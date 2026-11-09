"""Symbolic regression via rich feature basis + multi-strategy ridge + fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 295.0  # safety margin under 300s


def _build_features(X, max_p=14):
    """Build a rich feature matrix from input X (n, d)."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n, dtype=np.float64)]
    
    for j in range(d):
        x = X[:, j].astype(np.float64)
        ax = np.abs(x)
        
        # Polynomials x^1 .. x^max_p
        powv = x.copy()
        for p in range(1, max_p + 1):
            cols.append(powv.copy())
            if p < max_p:
                powv = powv * x
        
        # Absolute value and its powers
        cols.append(ax)
        cols.append(ax ** 0.5)
        cols.append(ax ** 0.25)
        cols.append(ax ** 1.5)
        cols.append(ax ** 2.5)
        
        # Signed fractional powers
        s = np.sign(x)
        cols.append(s * (ax ** 0.75))
        cols.append(s * (ax ** 1.25))
        cols.append(s * (ax ** 0.5))
        cols.append(s * (ax ** 1.5))
        cols.append(s * (ax ** 2.5))
        
        # Reciprocal terms (guarded)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(ax > 1e-8, 1.0 / x, 0.0)
            inv2 = np.where(ax > 1e-8, 1.0 / (x * x), 0.0)
        cols.append(inv)
        cols.append(inv2)
        cols.append(s / (ax + 1e-8))
        
        # Trigonometric
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.sin(ax))
        cols.append(np.cos(ax))
        cols.append(np.sin(x * 0.5))
        cols.append(np.cos(x * 0.5))
        cols.append(np.sin(x * 2.0))
        cols.append(np.cos(x * 2.0))
        cols.append(np.sin(np.sqrt(ax)))
        cols.append(np.cos(np.sqrt(ax)))
        
        # Exponential (clipped)
        xc = np.clip(x, -40, 40)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        cols.append(np.exp(-0.5 * x * x))
        cols.append(np.exp(-x * x))
        cols.append(np.exp(x * x * 0.25))
        cols.append(np.exp(-x * x * 0.25))
        
        # Logarithmic (guarded)
        cols.append(np.where(x > 1e-9, np.log(x), 0.0))
        cols.append(np.where(ax > 1e-9, np.log(ax), 0.0))
        cols.append(np.log1p(ax))
        cols.append(np.log1p(np.exp(-ax)))  # softplus-like
        
        # Hyperbolic (clipped)
        sc = np.clip(xc, -15, 15)
        cols.append(np.sinh(sc))
        cols.append(np.cosh(sc))
        with np.errstate(over='ignore', divide='ignore'):
            sech = 1.0 / np.cosh(sc)
            sech[~np.isfinite(sech)] = 0.0
        cols.append(sech)
        
        # Spline-like basis (piecewise linear bumps)
        for center in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            cols.append(np.maximum(0, x - center))
            cols.append(np.maximum(0, -x - center))
            cols.append(np.maximum(0, 2.0 - np.abs(x - center)))
    
    # Pairwise interactions for multi-dim
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(X[:, i] * X[:, j] * X[:, i])
                cols.append(X[:, i] * X[:, j] * X[:, j])
                cols.append(np.sin(X[:, i] * X[:, j]))
                cols.append(np.cos(X[:, i] * X[:, j]))
    
    Phi = np.column_stack(cols)
    return Phi


def _clean_features(Phi, y, var_th=1e-12):
    """Remove non-finite columns and zero-variance columns."""
    finite_cols = np.all(np.isfinite(Phi), axis=0)
    Phi = Phi[:, finite_cols]
    if Phi.shape[1] == 0:
        return Phi
    col_var = np.var(Phi, axis=0)
    keep = col_var > var_th
    Phi = Phi[:, keep]
    return Phi


def _ridge_solve(Phi, y, lam):
    """Solve ridge regression (Phi.T Phi + lam I) w = Phi.T y."""
    n_feat = Phi.shape[1]
    if lam == 0.0:
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    else:
        A = Phi.T @ Phi + lam * np.eye(n_feat)
        b = Phi.T @ y
        try:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L.T, np.linalg.solve(L, b))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w


def _nrmse(y, yhat):
    """Normalized MSE: mean((y-yhat)^2) / var(y)."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-15:
        return mse
    return mse / vy


def fit(X_train, y_train):
    t0 = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    n, d = X_train.shape
    y_mean = float(np.mean(y_train))
    y_var = float(np.var(y_train))
    
    # Build features
    try:
        Phi = _build_features(X_train, max_p=14)
        Phi = _clean_features(Phi, y_train)
    except Exception:
        # Fallback: simple polynomial
        cols = [np.ones(n)]
        for j in range(d):
            x = X_train[:, j]
            for p in range(1, 6):
                cols.append(x ** p)
        Phi = np.column_stack(cols)
    
    if Phi.shape[1] == 0:
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    # Standardize features for better conditioning
    Phi_mean = np.mean(Phi, axis=0)
    Phi_std = np.std(Phi, axis=0)
    Phi_std[Phi_std < 1e-12] = 1.0
    Phi_std = np.where(Phi_std < 1e-12, 1.0, Phi_std)
    Phi_norm = (Phi - Phi_mean) / Phi_std
    
    # Try multiple ridge lambdas
    best_w = None
    best_nrmse = np.inf
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for lam in lambdas:
        if time.time() - t0 > DEADLINE - 10:
            break
        try:
            w = _ridge_solve(Phi_norm, y_train, lam)
            if not np.all(np.isfinite(w)):
                continue
            yhat = Phi_norm @ w
            if not np.all(np.isfinite(yhat)):
                continue
            nm = _nrmse(y_train, yhat)
            if nm < best_nrmse:
                best_nrmse = nm
                best_w = w
                best_lam = lam
        except Exception:
            continue
    
    # If no good ridge found, try least squares directly
    if best_w is None:
        try:
            best_w, *_ = np.linalg.lstsq(Phi_norm, y_train, rcond=None)
            best_lam = 0.0
        except Exception:
            return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    # Try to improve with a sparse L1-ish approach via iterative reweighted least squares
    if time.time() - t0 < DEADLINE - 30:
        try:
            w_sparse = np.zeros_like(best_w)
            yhat_init = Phi_norm @ best_w
            resid = y_train - yhat_init
            for it in range(5):
                if time.time() - t0 > DEADLINE - 15:
                    break
                # Weights based on current solution
                wts = np.abs(best_w) + 1e-6
                # Solve weighted ridge
                try:
                    A = Phi.T @ Phi + 1e-4 * np.eye(Phi.shape[1])
                    b = Phi.T @ y_train
                    # Simple L1 approximation: soft-threshold
                    w_new = np.linalg.lstsq(A, b, rcond=None)[0]
                    # Soft threshold
                    thresh = 0.01 * np.std(y_train)
                    w_new = np.sign(w_new) * np.maximum(np.abs(w_new) - thresh, 0)
                    yhat_s = Phi_norm @ w_new
                    if np.all(np.isfinite(yhat_s)):
                        nm_s = _nrmse(y_train, yhat_s)
                        if nm_s < best_nrmse * 0.95:  # significant improvement
                            best_w = w_new
                            best_nrmse = nm_s
                except Exception:
                    break
        except Exception:
            pass
    
    # Final predict function
    Phi_mean_f = Phi_mean
    Phi_std_f = Phi_std
    best_w_f = best_w
    y_mean_f = y_mean
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        n_x = X.shape[0]
        try:
            P = _build_features(X, max_p=14)
            # Clean columns that might have issues
            finite_cols = np.all(np.isfinite(P), axis=0)
            if not np.all(finite_cols):
                P = P[:, finite_cols]
            # Match column count
            if P.shape[1] != Phi_mean_f.shape[0]:
                # Rebuild with same processing
                P_clean = _clean_features(P, np.zeros(P.shape[0]))
                if P_clean.shape[1] != Phi_mean_f.shape[0]:
                    return np.full(n_x, y_mean_f, dtype=np.float64)
                P = P_clean
            P_norm = (P - Phi_mean_f) / Phi_std_f
            yhat = P_norm @ best_w_f
            yhat = np.where(np.isfinite(yhat), yhat, y_mean_f)
            return yhat.astype(np.float64)
        except Exception:
            return np.full(n_x, y_mean_f, dtype=np.float64)
    
    return predict

# EVOLVE-BLOCK-END
