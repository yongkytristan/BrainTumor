import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Tumor Otak",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🧠 Klasifikasi Tumor Otak Menggunakan YOLOv8")
st.markdown("""
Website ini memprediksi jenis tumor otak (glioma, meningioma, notumor, pituitary)
dari gambar MRI yang diunggah atau diambil langsung dari kamera, menggunakan model YOLOv8 yang telah dilatih.
""")

model_path = 'best.pt'

@st.cache_resource
def load_yolov8_model(path):
    """Memuat model YOLOv8 dari path yang diberikan."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("Pastikan file model ada di path yang benar.")
        return None

model = load_yolov8_model(model_path)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary'] 

if model:
    st.success("Model YOLOv8 berhasil dimuat!")
else:
    st.warning("Model tidak dapat dimuat. Silakan periksa model kembali!")

st.header("Pilih Sumber Gambar")

# Inisialisasi session state untuk melacak pilihan pengguna
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("Ambil Gambar dari Kamera 📸"):
        st.session_state.input_mode = 'camera'
with col_btn2:
    if st.button("Unggah File Gambar 📁"):
        st.session_state.input_mode = 'upload'

image_to_predict = None
display_caption = ""

if st.session_state.input_mode == 'camera':
    st.subheader("Ambil Gambar dari Kamera")
    camera_image = st.camera_input("Klik 'Ambil Foto' untuk mengambil gambar")
    if camera_image is not None:
        image_to_predict = Image.open(camera_image)
        display_caption = 'Gambar dari Kamera'
elif st.session_state.input_mode == 'upload':
    st.subheader("Unggah File Gambar")
    uploaded_file = st.file_uploader("Pilih gambar dari file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        display_caption = 'Gambar yang Diunggah'
else:
    st.info("Silakan pilih apakah Anda ingin mengambil gambar dari kamera atau mengunggah file.")

if image_to_predict is not None:
    st.image(image_to_predict, caption=display_caption, use_column_width=True)
    st.write("")
    st.write("Gambar siap untuk diprediksi...")

    #Predict
    if st.button("Lakukan Prediksi"):
        if model is not None:
            try:
                results = model.predict(source=image_to_predict, imgsz=224, verbose=False)

                if results:
                    probs = results[0].probs
                    if probs is not None:
                        top_prob_index = probs.top1
                        confidence = probs.top1conf.cpu().numpy()
                        predicted_class_name = class_names[top_prob_index]

                        st.subheader("Hasil Prediksi:")
                        st.success(f"**Kelas Prediksi:** {predicted_class_name}")
                        st.info(f"**Tingkat Kepercayaan:** {confidence:.4f}")

                        # Tampilkan semua probabilitas (opsional)
                        st.subheader("Probabilitas untuk Setiap Kelas:")
                        prob_dict = {class_names[i]: probs.data[i].item() for i in range(len(class_names))}
                        st.write(prob_dict)

                    else:
                        st.error("Tidak ada probabilitas yang ditemukan dalam hasil prediksi.")
                else:
                    st.error("Tidak ada hasil prediksi yang diperoleh.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
                st.info("Pastikan gambar yang diunggah/diambil valid dan model dimuat dengan benar.")
        else:
            st.warning("Model belum dimuat. Tidak dapat melakukan prediksi.")
else:
    if st.session_state.input_mode is not None: # Hanya tampilkan ini jika sudah memilih mode tapi belum ada gambar
        st.info("Unggah gambar atau ambil dari kamera untuk memulai prediksi.")

st.markdown("---")
st.markdown("Kelompok 1 Machine Learning - LE01")