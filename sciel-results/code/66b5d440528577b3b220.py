<<<<<<< SEARCH
    w, v = eigh(A, subset_by_index=[0, band_idx])
    vec = v[:, band_idx]
=======
    w, v = np.linalg.eigh(A)
    vec = v[:, band_idx]
>>>>>>> REPLACE
