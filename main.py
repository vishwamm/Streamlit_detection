import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

model=YOLO('yolov8n.pt')
save1_dir='uploaded_images'
save2_dir='uploaded_videos'

st.title("OBJECT DETECTION USING YOLOV8:")
st.subheader("Select an image for detection:")
img=st.file_uploader("Image:")


if img is not None:
    st.success(f"Image uploaded successfully!")
    if st.button('DETECT'):
        imgname = img.name
        save1_path = os.path.join(save1_dir, imgname)
        if not os.path.exists(save1_dir):
            os.makedirs(save1_dir)
        with open(save1_path, "wb") as f:
            f.write(img.getbuffer())


        img="uploaded_images/"+imgname
        cap=cv2.imread(img)
        results=model(cap)
        annotated_frame=results[0].plot()
        cv2.imwrite(f"{img}.jpg", annotated_frame)
        out=f"{img}.jpg"
        st.subheader("Detected Image:")
        st.image(out)



st.subheader("Select a video for detection:")
video=st.file_uploader("Video:",type=["mp4"])

if video is not None:
    st.success("Video uploaded successfully!")
    if st.button('GENERATE'):
        videoname=video.name
        save2_path = os.path.join(save2_dir, videoname)
        if not os.path.exists(save2_dir):
            os.makedirs(save2_dir)
        with open(save2_path, "wb") as f:
            f.write(video.getbuffer())


        video_path="uploaded_videos/"+videoname
        output_video_filename = f"annotated_{videoname}.mp4"
        with st.spinner("Detection in process..."):
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'WebM')  # Adjust codec based on your needs
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_video_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = model(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame.squeeze().astype(np.uint8))
            cap.release()
            out.release()
            cv2.destroyAllWindows()
    
        st.video(output_video_filename)
        st.subheader("Click the play button in the detected video!")
    


