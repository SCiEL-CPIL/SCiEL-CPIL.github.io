try:
    L = np.linalg.cholesky(C)
except:
    # Add small regularization
    C += 1e-6 * np.eye(n)
    L = np.linalg.cholesky(C)
