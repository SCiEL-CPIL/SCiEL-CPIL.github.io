"""Symbolic regression with feature expansion + ensemble of linear models."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

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

def _build_feature_matrix(X, max_poly_degree=4, include_trig=True, include_exp_log=False):
    """
    Build a rich feature matrix from input X.
    Features include:
    - Original features
    - Polynomial terms up to max_poly_degree (all monomials)
    - Sin, cos of each feature
    - Tanh of each feature
    - If include_exp_log: exp(-x), log(x) for x>0
    """
    n, d = X.shape
    features = [np.ones((n, 1))]
    
    # Original features
    for j in range(d):
        features.append(X[:, j:j+1])
    
    # Polynomial monomials up to max_poly_degree
    # For d dimensions, generate all monomials x1^a1 * x2^a2 * ... * xd^ad where sum(ai) <= max_poly_degree
    # To keep it manageable, we limit the total degree
    if max_poly_degree >= 2:
        # Generate all combinations of exponents
        # For small d (<=3), we can enumerate
        for total_deg in range(2, max_poly_degree + 1):
            # Generate all compositions of total_deg into d parts (including zeros)
            # Use recursion or itertools
            def gen_compositions(total, parts):
                if parts == 1:
                    yield (total,)
                else:
                    for i in range(total + 1):
                        for rest in gen_compositions(total - i, parts - 1):
                            yield (i,) + rest
            
            for comp in gen_compositions(total_deg, d):
                feat = np.ones((n, 1))
                for j in range(d):
                    if comp[j] > 0:
                        feat = feat * (X[:, j:j+1] ** comp[j])
                features.append(feat)
    
    # Trigonometric features
    if include_trig:
        for j in range(d):
            features.append(np.sin(X[:, j:j+1]))
            features.append(np.cos(X[:, j:j+1]))
            features.append(np.tanh(X[:, j:j+1]))
    
    # Exponential/log features (careful with domain)
    if include_exp_log:
        for j in range(d):
            features.append(np.exp(-X[:, j:j+1]))
            # log only for positive values
            x_pos = np.maximum(X[:, j], 1e-10)
            features.append(np.log(x_pos))
    
    return np.hstack(features)

def _ridge_solve(A, b, lam):
    """Solve (A + lam*I) @ w = b using Cholesky or SVD fallback."""
    n = A.shape[0]
    try:
        # Try Cholesky for speed
        A_reg = A + lam * np.eye(n)
        L = np.linalg.cholesky(A_reg)
        # Solve L @ y = b
        y = np.linalg.solve(L, b)
        # Solve L.T @ w = y
        w = np.linalg.solve(L.T, y)
        return w
    except np.linalg.LinAlgError:
        try:
            # Fall back to SVD
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            S_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
            w = Vt.T @ (S_inv * (U.T @ b))
            return w
        except Exception:
            return np.zeros(n)

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
    
    # Build feature matrix
    # Start with a moderate feature set
    max_poly_degree = 3 if d <= 2 else 2
    F = _build_feature_matrix(X_norm, max_poly_degree=max_poly_degree, include_trig=True, include_exp_log=False)
    
    # If feature matrix is too large, reduce
    n_features = F.shape[1]
    if n_features > 500:
        # Fall back to simpler features
        F = _build_feature_matrix(X_norm, max_poly_degree=2, include_trig=True, include_exp_log=False)
        n_features = F.shape[1]
    
    # Split for validation
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    val_size = max(20, int(0.1 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    F_tr = F[train_idx]
    y_tr = y_norm[train_idx]
    F_val = F[val_idx]
    y_val = y_norm[val_idx]
    
    # Try different regularization strengths
    ridge_lambdas = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    best_val_mse = np.inf
    best_lam = None
    best_w = None
    
    # Precompute F_tr.T @ F_tr for efficiency
    FtrT_Ftr = F_tr.T @ F_tr
    
    for lam in ridge_lambdas:
        if _remaining() < 5.0:
            break
        try:
            w = _ridge_solve(FtrT_Ftr, F_tr.T @ y_tr, lam)
            pred_val = F_val @ w
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_lam = lam
                best_w = w
        except Exception:
            continue
    
    # Also try with more features if time permits
    if _remaining() > 30.0 and best_val_mse > 0.1:
        # Try with higher polynomial degree
        F2 = _build_feature_matrix(X_norm, max_poly_degree=4, include_trig=True, include_exp_log=False)
        if F2.shape[1] <= 1000:
            F2_tr = F2[train_idx]
            F2_val = F2[val_idx]
            F2trT_F2tr = F2_tr.T @ F2_tr
            
            for lam in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                if _remaining() < 5.0:
                    break
                try:
                    w2 = _ridge_solve(F2trT_F2tr, F2_tr.T @ y_tr, lam)
                    pred_val = F2_val @ w2
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_lam = lam
                        best_w = w2
                        # Use F2 as the feature matrix
                        F = F2
                except Exception:
                    continue
    
    # Final fit on ALL data with the best configuration
    F_full = F
    try:
        F_fullT_F_full = F_full.T @ F_full
        final_w = _ridge_solve(F_fullT_F_full, F_full.T @ y_norm, best_lam)
    except Exception:
        final_w = best_w
    
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_mean) / X_std
        F_t = _build_feature_matrix(X_t_norm, max_poly_degree=max_poly_degree, include_trig=True, include_exp_log=False)
        # Match feature count
        if F_t.shape[1] != F_full.shape[1]:
            # This shouldn't happen, but handle gracefully
            min_feat = min(F_t.shape[1], F_full.shape[1])
            F_t = F_t[:, :min_feat]
            w_use = final_w[:min_feat]
        else:
            w_use = final_w
        y_pred_norm = F_t @ w_use
        return y_pred_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
