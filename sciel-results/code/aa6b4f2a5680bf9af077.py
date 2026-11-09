"""Symbolic regression with adaptive basis selection and cross-validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    # Degree 2
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Degree 3 (only if degree >= 3)
    if degree >= 3:
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

def _hybrid_features(X, n_freqs=5):
    """Hybrid: linear + quadratic + cubic pure + sin/cos at multiple frequencies."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Linear
    for i in range(d):
        cols.append(X[:, i])
    # Quadratic
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Cubic pure powers
    for i in range(d):
        cols.append(X[:, i] ** 3)
    # Fourier
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

def _fit_ridge_best_lambda(X, y, lam_grid):
    """Fit ridge regression with best lambda from a grid via CV."""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    n_folds = 5
    fold_size = n // n_folds
    
    best_score = np.inf
    best_lam = lam_grid[0]
    
    for lam in lam_grid:
        mse_sum = 0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            w = _fit_ridge(X_tr, y_tr, lam)
            pred = _predict_ridge(X_val, w)
            if not np.all(np.isfinite(pred)):
                mse_sum = np.inf
                break
            mse_sum += np.mean((pred - y_val) ** 2)
        if mse_sum < best_score:
            best_score = mse_sum / n_folds
            best_lam = lam
    
    return _fit_ridge(X, y, best_lam), best_lam

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

def _cv_score_multi_lambda(X, y, feature_fn, n_folds=5, lam_grid=None):
    """Cross-validate a feature function with lambda sweep."""
    if lam_grid is None:
        lam_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    best_score = np.inf
    best_lam = lam_grid[0]
    
    for lam in lam_grid:
        mse_sum = 0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            try:
                F_train = feature_fn(X_train)
                F_val = feature_fn(X_val)
                w = _fit_ridge(F_train, y_train, lam)
                pred = _predict_ridge(F_val, w)
                if not np.all(np.isfinite(pred)):
                    mse_sum = np.inf
                    break
                mse_sum += np.mean((pred - y_val) ** 2)
            except Exception:
                mse_sum = np.inf
                break
        if mse_sum < best_score:
            best_score = mse_sum / n_folds
            best_lam = lam
    
    return best_score, best_lam

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
    
    # Lambda grid for ridge
    lam_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    # Candidate feature functions
    candidates = []
    
    # Polynomial degree 2
    candidates.append(('poly2', lambda Xn: _poly_features(Xn, 2)))
    
    # Polynomial degree 3
    if d <= 3:
        candidates.append(('poly3', lambda Xn: _poly_features(Xn, 3)))
    
    # RBF with different gammas (use more centers)
    rbf_centers = X_norm[::max(1, n // 20)]
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        candidates.append((f'rbf_g{gamma}', lambda Xn, g=gamma, c=rbf_centers: _rbf_features(Xn, c, g)))
    
    # Fourier features
    candidates.append(('fourier5', lambda Xn: _fourier_features(Xn, 5)))
    candidates.append(('fourier10', lambda Xn: _fourier_features(Xn, 10)))
    
    # Hybrid features
    candidates.append(('hybrid5', lambda Xn: _hybrid_features(Xn, 5)))
    
    # Evaluate candidates with cross-validation + lambda sweep
    best_score = np.inf
    best_fn = None
    best_name = None
    best_lam = 1e-6
    
    for name, fn in candidates:
        if time.time() > deadline - 40:
            break
        try:
            score, lam = _cv_score_multi_lambda(X_norm, y, fn, n_folds=5, lam_grid=lam_grid)
            if score < best_score:
                best_score = score
                best_fn = fn
                best_name = name
                best_lam = lam
        except Exception:
            continue
    
    # If no valid candidate, fall back to simple linear
    if best_fn is None:
        best_fn = lambda Xn: np.column_stack([np.ones(len(Xn)), Xn])
        best_name = 'linear'
        best_lam = 1e-6
    
    # Fit final model on all data with best lambda
    F = best_fn(X_norm)
    w = _fit_ridge(F, y, lam=best_lam)
    
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
