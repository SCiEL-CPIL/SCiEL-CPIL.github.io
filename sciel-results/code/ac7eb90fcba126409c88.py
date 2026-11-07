def _fit_loess(X, y, frac=0.5):
    """Fit LOESS model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    
    # Simple implementation of LOESS with local linear regression
    # For each test point, find nearest neighbors and fit a line
    # This is O(N^2) for prediction, which is fine for N=200.
    
    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        preds = np.zeros(len(xn))
        for i, xi in enumerate(xn):
            # Find k nearest neighbors
            dists = np.abs(x - xi)
            k = max(10, int(frac * n))
            idx = np.argsort(dists)[:k]
            x_loc = x[idx]
            y_loc = y[idx]
            
            # Weighted least squares for local linear model
            # Weights: tricube kernel
            h = dists[idx][-1]
            if h < 1e-10: h = 1.0
            u = dists[idx] / h
            w = (1.0 - u**3)**3
            w[w < 1e-10] = 1e-10
            
            A = np.column_stack([np.ones(len(x_loc)), x_loc])
            W = np.diag(w)
            try:
                # (A^T W A)^-1 A^T W y
                ATA = A.T @ W @ A
                ATy = A.T @ W @ y_loc
                coeffs = np.linalg.solve(ATA, ATy)
                preds[i] = coeffs[0] + coeffs[1] * xi
            except Exception:
                preds[i] = np.average(y_loc, weights=w)
        return preds

    return predict
