"""Symbolic regression via kernel ridge + basis portfolio with LOOCV selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 330.0
_SAFETY = 20.0


def _time_left():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _SAFETY
    return _DEADLINE - time.time()


def _loocv_ridge(B, y, lam):
    """Efficient LOOCV MSE for ridge regression using the hat-matrix formula."""
    n, p = B.shape
    A = B.T @ B + lam * np.eye(p)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    B_Ainv = B @ A_inv
    h = np.einsum('ij,ij->i', B, B_Ainv)
    yhat = B @ (A_inv @ (B.T @ y))
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    loocv = np.mean((resid / denom) ** 2)
    return loocv


def _loocv_kernel(K, y, lam):
    """Efficient LOOCV MSE for kernel ridge regression."""
    n = K.shape[0]
    A = K + lam * np.eye(n)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    K_Ainv = K @ A_inv
    h = np.einsum('ij,ij->i', K, K_Ainv)
    yhat = K @ (A_inv @ y)
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    loocv = np.mean((resid / denom) ** 2)
    return loocv


def _rbf_kernel(X1, X2, sigma):
    """Compute RBF kernel matrix between X1 and X2."""
    X1 = X1[:, 0:1]
    X2 = X2[:, 0:1]
    d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-d2 / (2 * sigma**2))


def _poly_kernel(X1, X2, degree, gamma=1.0):
    """Compute polynomial kernel matrix."""
    X1 = X1[:, 0:1]
    X2 = X2[:, 0:1]
    return (gamma * X1 @ X2.T + 1.0) ** degree


def _chebyshev_terms(x, max_deg):
    """Return list of Chebyshev polynomials T_0..T_max_deg evaluated at x."""
    n = x.shape[0]
    t0 = np.ones(n)
    t1 = x.copy()
    out = [t0, t1]
    for k in range(2, max_deg + 1):
        tk = 2.0 * x * t1 - t0
        out.append(tk)
        t0, t1 = t1, tk
    return out


def _build_basis(x, max_poly, n_fourier):
    """Build a rich basis for 1-D input x."""
    n = x.shape[0]
    cols = [np.ones(n)]
    for t in _chebyshev_terms(x, max_poly):
        cols.append(t)
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def fit(X_train, y_train):
    global _DEADLINE
    x = X_train[:, 0].astype(np.float64)
    y = y_train.astype(np.float64)
    n = x.shape[0]

    # Determine input range
    lo, hi = float(np.min(x)), float(np.max(x))
    span = max(hi - lo, 1e-6)

    # Candidate models: (name, kernel_type, params)
    # We'll evaluate each with LOOCV and pick the best
    best_model = None
    best_cv = np.inf
    best_predict = None

    # Time budget for model selection
    t_start = time.time()
    time_budget = 250.0  # Leave 80s for final computation and safety

    # --- Model 1: Kernel Ridge with RBF ---
    # Try several sigma values
    sigma_candidates = [span * s for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]
    lam_candidates = np.logspace(-6, 4, 15)

    for sigma in sigma_candidates:
        if time.time() - t_start > time_budget:
            break
        K = _rbf_kernel(X_train, X_train, sigma)
        for lam in lam_candidates:
            if time.time() - t_start > time_budget:
                break
            cv = _loocv_kernel(K, y, lam)
            if cv < best_cv:
                best_cv = cv
                # Store parameters
                best_model = ('rbf', sigma, lam)
                # Precompute solution for prediction
                A = K + lam * np.eye(n)
                try:
                    alpha = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    continue

                def predict_rbf(X, K_train=K, alpha=alpha, sigma=sigma, x_train=x):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    K_test = _rbf_kernel(X, X_train, sigma)
                    out = K_test @ alpha
                    return np.where(np.isfinite(out), out, 0.0)

                best_predict = predict_rbf

    # --- Model 2: Kernel Ridge with Polynomial ---
    for degree in [2, 3, 4, 5]:
        if time.time() - t_start > time_budget:
            break
        gamma = 1.0 / span
        K = _poly_kernel(X_train, X_train, degree, gamma)
        for lam in lam_candidates[:8]:  # Fewer lambda values for poly
            if time.time() - t_start > time_budget:
                break
            cv = _loocv_kernel(K, y, lam)
            if cv < best_cv:
                best_cv = cv
                best_model = ('poly', degree, gamma, lam)
                A = K + lam * np.eye(n)
                try:
                    alpha = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    continue

                def predict_poly(X, K_train=K, alpha=alpha, degree=degree, gamma=gamma, x_train=x):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    K_test = _poly_kernel(X, X_train, degree, gamma)
                    out = K_test @ alpha
                    return np.where(np.isfinite(out), out, 0.0)

                best_predict = predict_poly

    # --- Model 3: Basis Ridge (Chebyshev + Fourier) ---
    max_poly = 20
    n_fourier = 10
    B = _build_basis(x, max_poly, n_fourier)
    p = B.shape[1]
    for lam in lam_candidates:
        if time.time() - t_start > time_budget:
            break
        cv = _loocv_ridge(B, y, lam)
        if cv < best_cv:
            best_cv = cv
            best_model = ('basis', max_poly, n_fourier, lam)
            A = B.T @ B + lam * np.eye(p)
            try:
                w = np.linalg.solve(A, B.T @ y)
            except np.linalg.LinAlgError:
                continue

            def predict_basis(X, B_train=B, w=w, max_poly=max_poly, n_fourier=n_fourier):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                Bx = _build_basis(X[:, 0], max_poly, n_fourier)
                out = Bx @ w
                return np.where(np.isfinite(out), out, 0.0)

            best_predict = predict_basis

    # Fallback: simple linear regression if nothing worked
    if best_predict is None:
        def predict_linear(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # Fallback to mean
            return np.full(X.shape[0], np.mean(y))
        best_predict = predict_linear

    return best_predict
# EVOLVE-BLOCK-END
