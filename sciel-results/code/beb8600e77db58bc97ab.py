"""Symbolic regression using Anisotropic Kernel Ridge Regression with Random Search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _anisotropic_rbf_kernel(X1, X2, length_scales, variance):
    """
    Anisotropic RBF kernel.
    X1: (n, d), X2: (m, d)
    length_scales: (d,) array of length scales for each dimension.
    """
    # Normalize inputs to [0, 1] range is handled outside or assumed. 
    # Here we assume X is already normalized to [0,1] for better scaling of LS.
    # Distance squared: sum_j (X1_ij - X2_kj)^2 / (l_j^2)
    # Using broadcasting: 
    # X1: (n, 1, d), X2: (1, m, d)
    X1_exp = X1[:, np.newaxis, :]
    X2_exp = X2[np.newaxis, :, :]
    
    # Squared distances divided by length scale squared
    # l_sq: (1, 1, d)
    l_sq = length_scales[np.newaxis, np.newaxis, :] ** 2
    sq_dists = np.sum((X1_exp - X2_exp) ** 2 / l_sq, axis=2)
    
    return variance * np.exp(-0.5 * sq_dists)


def _fit_krr(X_train, y_train, n_candidates=50, seed=42):
    """
    Fit Kernel Ridge Regression with Anisotropic RBF + Linear kernel.
    Uses Random Search for hyperparameters: log_var, log_noise, log_ls (per dim).
    """
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_centered = (y_train - y_mean) / y_std
    
    # Normalize inputs to [0, 1]
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Hyperparameters to search:
    # log_var (kernel variance)
    # log_noise (regularization/noise)
    # log_ls (length scales, one per dimension)
    
    # Define search ranges in log space
    # Var: usually around 1.0, range [0.1, 10] -> log [-2.3, 2.3]
    # Noise: usually small, range [1e-6, 1e-1] -> log [-13.8, -2.3]
    # LS: usually around 0.1-0.5, range [0.05, 1.0] -> log [-3, 0]
    
    rng = np.random.RandomState(seed)
    
    best_score = np.inf
    best_params = None
    
    # Pre-compute linear kernel part for speed? 
    # K_total = K_rbf + K_linear
    # K_linear = X_norm @ X_norm.T
    
    K_lin = X_norm @ X_norm.T
    
    for _ in range(n_candidates):
        log_var = rng.uniform(-2.0, 2.0)
        log_noise = rng.uniform(-12.0, -2.0)
        log_ls = rng.uniform(-3.0, 0.0, size=d)
        
        var = np.exp(log_var)
        noise = np.exp(log_noise)
        ls = np.exp(log_ls)
        
        K_rbf = _anisotropic_rbf_kernel(X_norm, X_norm, ls, var)
        K = K_rbf + K_lin + noise * np.eye(n)
        
        try:
            # Solve (K + I*noise) w = y
            # Using Cholesky for stability
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
            
            # Cross-validation score? 
            # For speed in random search, we can use LOO-CV or just a holdout.
            # Since N=200, LOO is cheap: LOO_i = y_i - (K^{-1})_{ii} y_i ? No, that's for standard KRR.
            # Actually, for KRR, LOO prediction error can be computed from the solve.
            # Let's use a simple 5-fold CV to be safe and accurate for selection.
            
            # To save time, we can just evaluate on a small holdout or use LOO approximation.
            # Let's use LOO-CV which is exact for KRR and O(N^2) after solve.
            # K_inv_diag = diag(inv(K))
            # LOO_i = y_i - K_inv_diag[i] * (K @ y)_i ? 
            # Standard formula: LOO_i = y_i - (K^{-1} y)_i / (K^{-1})_{ii}
            
            # We have alpha = K^{-1} y_centered.
            # K_inv_diag is needed.
            # Computing full inverse is O(N^3). We already did Cholesky.
            # We can solve K x = e_i for each i? Too slow.
            # Better: Use the fact that K^{-1} = L^{-T} L^{-1}.
            # (K^{-1})_{ii} = || L^{-T} e_i ||^2.
            # This requires N solves of triangular systems, which is fast.
            
            K_inv_diag = np.zeros(n)
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                v = np.linalg.solve(L.T, e_i) # L.T v = e_i
                K_inv_diag[i] = np.dot(v, v)
            
            # LOO predictions
            loo_preds = y_centered - alpha / K_inv_diag
            loo_mse = np.mean((loo_preds - y_centered) ** 2)
            
            if loo_mse < best_score:
                best_score = loo_mse
                best_params = (log_var, log_noise, log_ls)
                
        except np.linalg.LinAlgError:
            continue
    
    if best_params is None:
        # Fallback to reasonable defaults
        log_var = 0.0
        log_noise = -5.0
        log_ls = np.zeros(d)
    else:
        log_var, log_noise, log_ls = best_params
    
    var = np.exp(log_var)
    noise = np.exp(log_noise)
    ls = np.exp(log_ls)
    
    K_rbf = _anisotropic_rbf_kernel(X_norm, X_norm, ls, var)
    K = K_rbf + K_lin + noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    
    def predict_krr(X):
        X_min_ = X_train.min(axis=0, keepdims=True)
        X_max_ = X_train.max(axis=0, keepdims=True)
        X_range_ = X_max_ - X_min_
        X_range_[X_range_ < 1e-10] = 1.0
        X_norm_ = (X - X_min_) / X_range_
        
        K_rbf_test = _anisotropic_rbf_kernel(X_norm_, X_norm, ls, var)
        K_lin_test = X_norm_ @ X_norm.T
        
        K_test = K_rbf_test + K_lin_test
        
        preds = K_test @ alpha * y_std + y_mean
        return preds
    
    return predict_krr


def _fit_polynomial_fallback(X_train, y_train, degree=3):
    """Fallback polynomial regression if KRR fails."""
    n, d = X_train.shape
    
    def gen_features(X, deg):
        features = [np.ones(len(X))]
        for j in range(d):
            features.append(X[:, j])
        for total_deg in range(2, deg + 1):
            for i in range(d):
                for j in range(i, d):
                    if i == j:
                        features.append(X[:, i] ** total_deg)
                    else:
                        if total_deg == 2:
                            features.append(X[:, i] * X[:, j])
                        elif total_deg == 3:
                            # Avoid too many features
                            if i + j < d: 
                                features.append(X[:, i] * X[:, j] * X[:, max(i, j)])
        return np.column_stack(features)
    
    F = gen_features(X_train, 3)
    # Ridge
    alpha_ridge = 1e-3
    w = np.linalg.lstsq(F + alpha_ridge * np.eye(F.shape[1]), y_train, rcond=None)[0]
    
    def predict_poly(X):
        F_test = gen_features(X, 3)
        return F_test @ w
    
    return predict_poly


def fit(X_train, y_train):
    """
    Fit using Anisotropic Kernel Ridge Regression.
    """
    n, d = X_train.shape
    
    # Try KRR first
    try:
        predict_krr = _fit_krr(X_train, y_train, n_candidates=50)
        # Sanity check: ensure predictions are finite
        test_pred = predict_krr(X_train)
        if np.all(np.isfinite(test_pred)):
            return predict_krr
    except Exception:
        pass
    
    # Fallback to Polynomial
    try:
        return _fit_polynomial_fallback(X_train, y_train, degree=3)
    except Exception:
        # Last resort: Mean
        y_mean = np.mean(y_train)
        def predict_mean(X):
            return np.full(len(X), y_mean)
        return predict_mean
# EVOLVE-BLOCK-END
