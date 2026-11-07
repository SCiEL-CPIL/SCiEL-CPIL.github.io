"""Symbolic regression with rich feature expansion, kernel ensembles, and adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import pearsonr

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _rbf_kernel(X1, X2, length_scale=1.0):
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * length_scale**2))

def _poly_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def _linear_kernel(X1, X2):
    return X1 @ X2.T

def _fit_krr(X, y, kernel_fn, alpha, lam=1e-6):
    K = kernel_fn(X, X)
    n = K.shape[0]
    A = K + lam * np.eye(n)
    b = y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _cv_score_krr(X, y, kernel_fn, alpha, lam=1e-6, n_folds=5):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    mse_sum = 0.0
    for i in range(n_folds):
        val_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        w = _fit_krr(X_train, y_train, kernel_fn, alpha, lam)
        K_val = kernel_fn(X_val, X_train)
        pred = K_val @ w
        mse = np.mean((pred - y_val)**2)
        mse_sum += mse
    return mse_sum / n_folds

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Build rich feature matrix
    def build_features(X):
        n = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n)]
        
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        
        # Polynomial terms up to degree 4
        for deg in range(2, 5):
            _poly_monomials = []
            def _gen(idx, remaining, current):
                if idx == d - 1:
                    _poly_monomials.append(current + [remaining])
                    return
                for k in range(remaining + 1):
                    _gen(idx + 1, remaining - k, current + [k])
            _gen(0, deg, [])
            for exps in _poly_monomials:
                term = np.ones(n)
                for j in range(d):
                    if exps[j] > 0:
                        term = term * np.power(X[:, j], exps[j])
                cols.append(term)
        
        # Trigonometric terms: sin, cos of each input
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
        
        # sin/cos of sums of pairs
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                cols.append(np.cos(X[:, i] + X[:, j]))
                cols.append(np.sin(X[:, i] - X[:, j]))
                cols.append(np.cos(X[:, i] - X[:, j]))
        
        # Exponential and log terms
        for j in range(d):
            cols.append(np.exp(X[:, j] / 5.0))
            cols.append(np.log(X[:, j] + 1.0))
        
        # Products of pairs
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(np.sqrt(np.maximum(X[:, i] * X[:, j], 0)))
        
        # Ratio terms (safe)
        for i in range(d):
            for j in range(i + 1, d):
                denom = X[:, j] + 1e-10
                cols.append(X[:, i] / denom)
        
        return np.column_stack(cols)
    
    F = build_features(X)
    n_features = F.shape[1]
    
    # Normalize features for better conditioning
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Try Lasso-style feature selection using iterative hard thresholding
    # Start with ridge, then prune
    
    def ridge_solve(F_mat, y_vec, lam):
        n_feat = F_mat.shape[1]
        A = F_mat.T @ F_mat + lam * np.eye(n_feat)
        b = F_mat.T @ y_vec
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w
    
    # Try different lambda values
    best_score = np.inf
    best_w = None
    
    # Use a subset of lambdas
    lam_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    for lam in lam_values:
        w = ridge_solve(F_norm, y_norm, lam)
        pred = F_norm @ w
        mse = np.mean((pred - y_norm) ** 2)
        if mse < best_score:
            best_score = mse
            best_w = w
    
    # Now try L1-style pruning: start from ridge solution, zero out small coefficients
    # and refit
    w_current = best_w.copy()
    for iteration in range(5):
        mask = np.abs(w_current) > 0.01
        if not np.any(mask):
            break
        F_active = F_norm[:, mask]
        try:
            w_active, *_ = np.linalg.lstsq(F_active, y_norm, rcond=None)
        except:
            break
        w_new = np.zeros_like(w_current)
        w_new[mask] = w_active
        pred_new = F_norm @ w_new
        mse_new = np.mean((pred_new - y_norm) ** 2)
        if mse_new < best_score * 1.01:
            best_score = mse_new
            best_w = w_new
            w_current = w_new
        else:
            break
    
    # Final: also try pure least squares on all features as fallback
    try:
        w_ls, *_ = np.linalg.lstsq(F_norm, y_norm, rcond=None)
        pred_ls = F_norm @ w_ls
        mse_ls = np.mean((pred_ls - y_norm) ** 2)
        if mse_ls < best_score:
            best_w = w_ls
            best_score = mse_ls
    except:
        pass
    
    # Build prediction function from feature-based model
    def predict_feature(X_test):
        X_t = X_test.astype(np.float64)
        F_t = build_features(X_t)
        F_t_norm = (F_t - F_mean) / F_std
        y_pred = F_t_norm @ best_w
        return y_pred * y_std + y_mean
    
    # Now try kernel ridge ensembles and pick the best
    best_cv_score = np.inf
    best_model = None
    best_model_type = 'feature'
    
    # Check time budget
    if _remaining() > 30:
        # Try RBF kernels with different length scales
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        y_for_krr = y.copy()
        
        rbf_length_scales = [0.5, 1.0, 2.0, 5.0]
        for ls in rbf_length_scales:
            if _remaining() < 15:
                break
            kernel_fn = lambda X1, X2, ls=ls: _rbf_kernel(X1, X2, length_scale=ls)
            cv_score = _cv_score_krr(X_scaled, y_for_krr, kernel_fn, alpha=1.0, lam=1e-6, n_folds=5)
            if cv_score < best_cv_score:
                best_cv_score = cv_score
                best_model = ('rbf', ls)
                best_model_type = 'kernel'
        
        # Try polynomial kernels
        for deg in [2, 3]:
            if _remaining() < 15:
                break
            kernel_fn = lambda X1, X2, deg=deg: _poly_kernel(X1, X2, degree=deg, gamma=1.0, coef0=1.0)
            cv_score = _cv_score_krr(X_scaled, y_for_krr, kernel_fn, alpha=1.0, lam=1e-6, n_folds=5)
            if cv_score < best_cv_score:
                best_cv_score = cv_score
                best_model = ('poly', deg)
                best_model_type = 'kernel'
        
        # Try linear kernel
        if _remaining() > 15:
            kernel_fn = lambda X1, X2: _linear_kernel(X1, X2)
            cv_score = _cv_score_krr(X_scaled, y_for_krr, kernel_fn, alpha=1.0, lam=1e-6, n_folds=5)
            if cv_score < best_cv_score:
                best_cv_score = cv_score
                best_model = ('linear', 1)
                best_model_type = 'kernel'
    
    # Decide which model to use
    if best_model_type == 'kernel' and best_model is not None:
        model_type, param = best_model
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        if model_type == 'rbf':
            kernel_fn = lambda X1, X2: _rbf_kernel(X1, X2, length_scale=param)
        elif model_type == 'poly':
            kernel_fn = lambda X1, X2: _poly_kernel(X1, X2, degree=param, gamma=1.0, coef0=1.0)
        else:
            kernel_fn = lambda X1, X2: _linear_kernel(X1, X2)
        
        w_krr = _fit_krr(X_scaled, y, kernel_fn, alpha=1.0, lam=1e-6)
        X_train_scaled = X_scaled
        
        def predict_kernel(X_test):
            X_t = X_test.astype(np.float64)
            X_t_scaled = (X_t - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
            K_val = kernel_fn(X_t_scaled, X_train_scaled)
            return K_val @ w_krr
        return predict_kernel
    else:
        return predict_feature
# EVOLVE-BLOCK-END
