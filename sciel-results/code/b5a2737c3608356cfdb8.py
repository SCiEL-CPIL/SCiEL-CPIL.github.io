"""Combined symbolic regression: rich basis + cross-validated ridge + spline basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_features(X, max_p=10):
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
        cols.append(np.sqrt(ax))
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
                cols.append(X[:, i] * X[:, i])
    
    Phi = np.column_stack(cols)
    return Phi


def _clean_features(Phi, var_th=1e-12):
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
    DEADLINE = 290.0  # safety margin under 300s
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    n, d = X_train.shape
    y_mean = float(np.mean(y_train))
    
    # Build features
    try:
        Phi = _build_features(X_train, max_p=10)
        Phi = _clean_features(Phi)
    except Exception:
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
    Phi_norm = (Phi - Phi_mean) / Phi_std
    
    # Cross-validation based model selection to prevent overfitting
    # Split into 5 folds
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(n_folds)]
    folds[-1] = indices[(n_folds - 1) * fold_size:]  # Handle remainder
    
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    best_cv_nrmse = np.inf
    best_lam = 1e-4
    
    for lam in lambdas:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        cv_scores = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            try:
                w_cv = _ridge_solve(Phi_norm[train_idx], y_train[train_idx], lam)
                yhat_cv = Phi_norm[val_idx] @ w_cv
                if np.all(np.isfinite(yhat_cv)):
                    cv_scores.append(_nrmse(y_train[val_idx], yhat_cv))
            except Exception:
                continue
        
        if len(cv_scores) > 0:
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv_nrmse:
                best_cv_nrmse = mean_cv
                best_lam = lam
    
    # Final fit with best lambda on all data
    try:
        best_w = _ridge_solve(Phi_norm, y_train, best_lam)
        if not np.all(np.isfinite(best_w)):
            best_w = np.linalg.lstsq(Phi_norm, y_train, rcond=None)[0]
    except Exception:
        m = float(np.mean(y_train))
        return lambda X: np.full(X.shape[0], m, dtype=np.float64)
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        n_test = X.shape[0]
        try:
            P = _build_features(X, max_p=10)
            P = _clean_features(P)
            P_norm = (P - Phi_mean) / Phi_std
            yhat = P_norm @ best_w
            yhat = np.where(np.isfinite(yhat), yhat, y_mean)
            return yhat.astype(np.float64)
        except Exception:
            return np.full(n_test, y_mean, dtype=np.float64)
    
    return predict

# EVOLVE-BLOCK-END
