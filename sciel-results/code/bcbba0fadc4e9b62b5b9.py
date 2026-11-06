    def _ridge_solve(V, y, lam):
        """Solve ridge regression using SVD for numerical stability."""
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
        Uy = U.T @ y
        denom = s**2 + lam
        coeff = Uy * (s / denom)
        w = Vt.T @ coeff
        return w
    