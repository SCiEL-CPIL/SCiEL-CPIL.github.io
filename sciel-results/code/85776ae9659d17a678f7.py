eigvals, eigvecs = np.linalg.eigh(A * dt)
y_end = eigvecs @ (np.exp(eigvals) * (eigvecs.T @ y0))
