X = np.einsum('ir,jr,kr->ijk', U, V, W)
T = np.ones((n, m, p))
residuals = (X - T).flatten()
