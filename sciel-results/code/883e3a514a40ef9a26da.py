"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _rbf_kernel(X, Y, length_scale):
    """Compute RBF kernel matrix between X and Y."""
    X2 = np.sum(X**2, axis=1)[:, None]
    Y2 = np.sum(Y**2, axis=1)[:, None].T
    dists = X2 + Y2 - 2 * X @ Y.T
    np.maximum(dists, 0, out=dists)
    return np.exp(-dists / (2 * length_scale**2))

def _fit_kernel_ridge(X, y, length_scale, alpha):
    """Kernel ridge regression with RBF kernel."""
    K = _rbf_kernel(X, X, length_scale)
    K_reg = K + alpha * np.eye(len(X))
    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    def predict(X_test):
        K_test = _rbf_kernel(X_test, X, length_scale)
        return K_test @ w
    return predict

def _fit_sparse_gp(X, y, n_inducing=50, length_scale=1.0, alpha=1e-6):
    """Sparse GP regression with inducing points."""
    n = len(X)
    d = X.shape[1]
    
    # Initialize inducing points using k-means-like approach
    # Use a subset of training points as inducing points
    if n_inducing >= n:
        Z = X.copy()
    else:
        # Random subset with some stratification
        indices = np.random.choice(n, n_inducing, replace=False)
        Z = X[indices].copy()
    
    K_ZZ = _rbf_kernel(Z, Z, length_scale)
    K_ZZ += alpha * np.eye(n_inducing)
    K_ZX = _rbf_kernel(Z, X, length_scale)
    
    try:
        L = np.linalg.cholesky(K_ZZ)
        L_inv = np.linalg.inv(L)
        V = L_inv @ K_ZX
        VtV = V.T @ V
        Vty = V.T @ y
        u = np.linalg.solve(VtV + alpha * np.eye(n_inducing), Vty)
    except np.linalg.LinAlgError:
        # Fallback to kernel ridge
        return _fit_kernel_ridge(X, y, length_scale, 1e-3)
    
    def predict(X_test):
        K_ZX_test = _rbf_kernel(Z, X_test, length_scale)
        K_XZ_test = K_ZX_test.T
        mean = K_XZ_test @ L_inv @ V @ u
        return mean
    
    return predict

def _fit_polynomial(X, y, degree=3):
    """Polynomial regression with specified degree."""
    d = X.shape[1]
    
    def make_features(X):
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(d):
            cols.append(X[:, j:j+1])
        # Interaction and higher order terms up to given degree
        for deg in range(2, degree + 1):
            for i1 in range(d):
                for i2 in range(i1, d):
                    if deg == 2:
                        cols.append((X[:, i1] * X[:, i2])[:, None])
                    elif deg == 3:
                        for i3 in range(i2, d):
                            cols.append((X[:, i1] * X[:, i2] * X[:, i3])[:, None])
        return np.column_stack(cols)
    
    F_train = make_features(X)
    try:
        w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
    except:
        w = np.zeros(F_train.shape[1])
    
    def predict(X_test):
        F_test = make_features(X_test)
        return F_test @ w
    
    return predict

def _fit_ensemble(X, y, deadline):
    """Try multiple fitting strategies and return the best one."""
    n = len(X)
    d = X.shape[1]
    start_time = time.time()
    
    # Normalize inputs for better kernel behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    candidates = []
    
    # Strategy 1: Kernel ridge with different length scales
    for ls in [0.5, 1.0, 2.0, 5.0]:
        if time.time() - start_time > deadline - 30:
            break
        try:
            pred_func = _fit_kernel_ridge(X_norm, y, ls, 1e-4)
            train_pred = pred_func(X_norm)
            mse = np.mean((train_pred - y)**2)
            candidates.append((mse, pred_func, X_mean, X_std, "kr_ls_%.1f" % ls))
        except Exception:
            pass
    
    # Strategy 2: Sparse GP
    for ls in [1.0, 2.0]:
        if time.time() - start_time > deadline - 30:
            break
        try:
            n_ind = min(50, n)
            pred_func = _fit_sparse_gp(X_norm, y, n_inducing=n_ind, length_scale=ls)
            train_pred = pred_func(X_norm)
            mse = np.mean((train_pred - y)**2)
            candidates.append((mse, pred_func, X_mean, X_std, "sgp_ls_%.1f" % ls))
        except Exception:
            pass
    
    # Strategy 3: Polynomial regression
    for deg in [2, 3]:
        if time.time() - start_time > deadline - 30:
            break
        try:
            pred_func = _fit_polynomial(X, y, degree=deg)
            train_pred = pred_func(X)
            mse = np.mean((train_pred - y)**2)
            candidates.append((mse, pred_func, None, None, "poly_deg_%d" % deg))
        except Exception:
            pass
    
    if not candidates:
        # Fallback: simple linear regression
        X_aug = np.column_stack([np.ones(n), X])
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_test):
            X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
            return X_test_aug @ w
        return predict
    
    # Pick the best candidate based on training MSE
    candidates.sort(key=lambda x: x[0])
    best_mse, best_pred, X_mean_c, X_std_c, name = candidates[0]
    
    if X_mean_c is not None:
        # Need to denormalize
        def final_predict(X_test):
            X_test_norm = (X_test - X_mean_c) / X_std_c
            return best_pred(X_test_norm)
    else:
        final_predict = best_pred
    
    return final_predict

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Deadline: leave 50s buffer for all 10 equations (300s total / 10 = 30s each, use 25s)
    per_eq_deadline = 25.0
    start_time = time.time()
    
    try:
        predict = _fit_ensemble(X_train, y_train, per_eq_deadline)
    except Exception:
        # Ultimate fallback: mean predictor
        y_mean = np.mean(y_train)
        predict = lambda X: np.full(len(X), y_mean)
    
    # Wrap to ensure finite outputs
    def safe_predict(X):
        X = np.asarray(X, dtype=np.float64)
        preds = predict(X)
        preds = np.asarray(preds, dtype=np.float64)
        # Replace any non-finite values with the training mean
        if not np.all(np.isfinite(preds)):
            y_mean = np.mean(y_train)
            preds = np.where(np.isfinite(preds), preds, y_mean)
        return preds
    
    return safe_predict
# EVOLVE-BLOCK-END
