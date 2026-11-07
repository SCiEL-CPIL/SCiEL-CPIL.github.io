"""Symbolic regression with kernel ensemble (RBF, Poly, Linear)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracker
_total_start = None
_time_limit = 310.0  # Leave 20s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix between X1 and X2."""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    np.maximum(sq_dists, 0, out=sq_dists)
    return np.exp(-gamma * sq_dists)

def _poly_kernel(X1, X2, degree, coef0):
    """Compute polynomial kernel matrix between X1 and X2."""
    return (X1 @ X2.T + coef0) ** degree

def _linear_kernel(X1, X2):
    """Compute linear kernel matrix between X1 and X2."""
    return X1 @ X2.T

def _krr_fit(X_tr, y_tr, kernel_fn, lam):
    """Fit Kernel Ridge Regression: solve (K + lam*I) w = y. Return (w, kernel_fn, X_tr)."""
    K = kernel_fn(X_tr, X_tr)
    n = K.shape[0]
    try:
        K_reg = K + lam * np.eye(n)
        w = np.linalg.solve(K_reg, y_tr)
        return w
    except np.linalg.LinAlgError:
        try:
            U, S, Vt = np.linalg.svd(K_reg, full_matrices=False)
            S_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
            w = Vt.T @ (S_inv * (U.T @ y_tr))
            return w
        except Exception:
            return np.zeros(n)

def _krr_predict(X_test, X_tr, w, kernel_fn):
    """Predict using Kernel Ridge Regression."""
    K_test = kernel_fn(X_test, X_tr)
    return K_test @ w

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Split for validation
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    val_size = max(20, int(0.15 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    X_tr = X_norm[train_idx]
    y_tr = y_norm[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_norm[val_idx]
    
    # Define kernel candidates with hyperparameters
    # RBF kernels with various gammas
    rbf_gammas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    # Polynomial kernels
    poly_params = [(2, 1.0), (2, 0.5), (3, 1.0), (3, 0.5), (4, 1.0)]
    # Linear kernel
    
    kernel_candidates = []
    for gamma in rbf_gammas:
        kernel_candidates.append(('rbf', lambda X1, X2, g=gamma: _rbf_kernel(X1, X2, g), gamma))
    for deg, c0 in poly_params:
        kernel_candidates.append(('poly', lambda X1, X2, d=deg, c=c0: _poly_kernel(X1, X2, d, c), (deg, c0)))
    kernel_candidates.append(('linear', lambda X1, X2: _linear_kernel(X1, X2), None))
    
    # Regularization strengths
    ridge_lams = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    best_val_mse = np.inf
    best_model = None  # (kernel_fn, w, X_tr_full, kernel_type)
    
    # For each kernel, find best lambda
    for kname, kfn, kparams in kernel_candidates:
        if _remaining() < 10.0:
            break
        best_lam_for_kernel = None
        best_w_for_kernel = None
        best_mse_for_kernel = np.inf
        
        for lam in ridge_lams:
            if _remaining() < 5.0:
                break
            try:
                w = _krr_fit(X_tr, y_tr, kfn, lam)
                pred_val = _krr_predict(X_val, X_tr, w, kfn)
                mse = np.mean((pred_val - y_val) ** 2)
                if mse < best_mse_for_kernel:
                    best_mse_for_kernel = mse
                    best_lam_for_kernel = lam
                    best_w_for_kernel = w
            except Exception:
                continue
        
        if best_lam_for_kernel is not None and best_mse_for_kernel < best_val_mse:
            best_val_mse = best_mse_for_kernel
            # Store the kernel function and parameters
            best_model = (kfn, best_lam_for_kernel, kname, kparams)
    
    if _remaining() < 5.0 or best_model is None:
        # Fallback: simple linear regression on normalized features
        try:
            # Add bias
            X_tr_b = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
            X_val_b = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
            w_lin = np.linalg.lstsq(X_tr_b, y_tr, rcond=None)[0]
            pred_val = X_val_b @ w_lin
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                # Use a linear kernel with bias handled by adding constant feature
                def _linear_bias_kernel(X1, X2):
                    X1_b = np.hstack([X1, np.ones((X1.shape[0], 1))])
                    X2_b = np.hstack([X2, np.ones((X2.shape[0], 1))])
                    return X1_b @ X2_b.T
                best_model = (_linear_bias_kernel, 1e-8, 'linear_bias', None)
        except Exception:
            pass
    
    # Final fit on ALL data with the best configuration
    if best_model is not None:
        kfn, best_lam, kname, kparams = best_model
        try:
            final_w = _krr_fit(X_norm, y_norm, kfn, best_lam)
            X_ref = X_norm
        except Exception:
            # Fallback to training subset
            final_w = _krr_fit(X_tr, y_tr, kfn, best_lam)
            X_ref = X_tr
        
        def predict(X_test):
            X_t = X_test.astype(np.float64)
            X_t_norm = (X_t - X_mean) / X_std
            try:
                y_pred_norm = _krr_predict(X_t_norm, X_ref, final_w, kfn)
            except Exception:
                # Fallback: predict y_mean
                y_pred_norm = np.full(X_t.shape[0], y_mean)
            return y_pred_norm * y_std + y_mean
    else:
        # Ultimate fallback: predict mean
        def predict(X_test):
            return np.full(X_test.shape[0], y_mean)
    
    return predict
# EVOLVE-BLOCK-END
