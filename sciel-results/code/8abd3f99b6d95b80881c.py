"""Symbolic regression with adaptive basis selection and cross-validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for i in range(d):
            for j in range(i, d):
                if deg == 1:
                    cols.append(X[:, i] ** deg)
                else:
                    # For degree > 1, generate all monomials of total degree = deg
                    pass
    # Simpler: just use all monomials up to degree
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    # Degree 2
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Degree 3
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _rbf_features(X, centers, gamma):
    """Generate RBF features."""
    n = X.shape[0]
    d = X.shape[1]
    # Normalize centers
    center_mean = centers.mean(axis=0)
    center_std = centers.std(axis=0) + 1e-10
    X_norm = (X - center_mean) / center_std
    C_norm = (centers - center_mean) / center_std
    
    # Compute distances
    diff = X_norm[:, np.newaxis, :] - C_norm[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * dist_sq)

def _fourier_features(X, n_freqs=5):
    """Generate Fourier features."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Normalize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    for freq in range(1, n_freqs + 1):
        for j in range(d):
            cols.append(np.sin(2 * np.pi * freq * X_norm[:, j]))
            cols.append(np.cos(2 * np.pi * freq * X_norm[:, j]))
    return np.column_stack(cols)

def _fit_ridge(X, y, lam=1e-6):
    """Fit ridge regression, return weights and intercept."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    b = Xb.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _predict_ridge(X, w):
    """Predict using ridge model."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    return Xb @ w

def _cv_score(X, y, feature_fn, n_folds=5, lam=1e-6):
    """Cross-validate a feature function."""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    mse_sum = 0
    
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        try:
            F_train = feature_fn(X_train)
            F_val = feature_fn(X_val)
            
            # Fit ridge
            w = _fit_ridge(F_train, y_train, lam)
            
            pred = _predict_ridge(F_val, w)
            if not np.all(np.isfinite(pred)):
                return np.inf
            
            mse = np.mean((pred - y_val) ** 2)
            mse_sum += mse
        except Exception:
            return np.inf
    
    return mse_sum / n_folds

def fit(X_train, y_train):
    """Fit a regression model using adaptive basis selection."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize for feature generation
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    # Candidate feature functions
    candidates = []
    
    # Polynomial degree 2
    candidates.append(('poly2', lambda Xn: _poly_features(Xn, 2)))
    
    # RBF with different gammas
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        candidates.append((f'rbf_g{gamma}', lambda Xn, g=gamma: _rbf_features(Xn, Xn[::10], g)))
    
    # Fourier features
    candidates.append(('fourier5', lambda Xn: _fourier_features(Xn, 5)))
    candidates.append(('fourier10', lambda Xn: _fourier_features(Xn, 10)))
    
    # Evaluate candidates with cross-validation
    best_score = np.inf
    best_fn = None
    best_name = None
    
    for name, fn in candidates:
        if time.time() > deadline - 30:
            break
        try:
            score = _cv_score(X_norm, y, fn, n_folds=5, lam=1e-6)
            if score < best_score:
                best_score = score
                best_fn = fn
                best_name = name
        except Exception:
            continue
    
    # If no valid candidate, fall back to simple linear
    if best_fn is None:
        best_fn = lambda Xn: np.column_stack([np.ones(len(Xn)), Xn])
        best_name = 'linear'
    
    # Fit final model on all data
    F = best_fn(X_norm)
    w = _fit_ridge(F, y, lam=1e-6)
    
    # Create predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test_norm = (X_test - X_mean) / X_std
        F_test = best_fn(X_test_norm)
        pred = _predict_ridge(F_test, w)
        # Ensure finite output
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred
    
    return predict
# EVOLVE-BLOCK-END
