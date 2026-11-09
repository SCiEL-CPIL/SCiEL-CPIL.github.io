import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 70 # 70s per equation to be safe (4 * 70 = 280 < 300)
    
    X = X_train.flatten()
    y = y_train.flatten()
    n = len(y)
    
    # Sort data for potential local methods, though GP doesn't strictly need it
    # But for LOESS or local regression, it helps.
    # Let's try a combination:
    # 1. Try to fit a GP with a simple length-scale search using LOO-CV (efficient for 1D)
    # 2. If that fails or is too slow, fallback to LOESS or Polynomial.
    
    # Actually, for 1D noiseless data, a simple approach that often works well is
    # to use a kernel with a length scale chosen via cross-validation, 
    # or just use a very flexible kernel (small length scale) but regularized.
    # Since data is noiseless, interpolation is ideal.
    # GP with small noise floor and optimized length scale.
    
    # Let's use a simpler optimization: Grid search or simple local optimization on length scale.
    # The signal variance is less critical if we normalize y.
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Kernel
    def rbf_kernel(X1, X2, ell):
        diff = X1[:, None] - X2[None, :]
        return np.exp(-0.5 * diff**2 / ell**2)
    
    # Objective: Leave-One-Out Cross-Validation Error
    # For GP, LOO predictions can be computed efficiently from the Cholesky decomposition.
    # y_i_loo = y_i - K_ii^-1 * alpha_i (where alpha = K^-1 y)
    # Specifically, if K = L L^T, alpha = L^-T L^-1 y.
    # The LOO error for point i is:
    # e_i = y_i - (K^-1 y)_i / (K^-1)_ii
    # Note: (K^-1)_ii = 1 / (L L^T)^-1_ii ... actually simpler:
    # Let v = solve(L, y). Then alpha = solve(L.T, v).
    # (K^-1)_ii = 1 / (L L^T)_ii? No.
    # (K^-1)_ii = 1 / (L L^T)^-1_ii is wrong.
    # (K^-1)_ii = 1 / (L L^T)_ii is wrong.
    # Let's use the formula: LOO_pred_i = y_i - K_ii^-1 alpha_i is not right.
    # Correct formula:
    # mu_loo_i = y_i - (K^-1 y)_i / (K^-1)_ii
    # where (K^-1)_ii is the i-th diagonal element of the inverse kernel matrix.
    # We can compute K_inv_diag efficiently from L:
    # K_inv_diag = 1 / np.diag(np.linalg.solve(L, np.linalg.solve(L.T, np.eye(n))))
    # This is O(N^3) which is fine for N=200.
    
    def loo_cv_error(ell):
        if time.time() > deadline:
            return np.inf
        K = rbf_kernel(X, X, ell)
        noise = 1e-8 # Small noise for stability
        K[np.diag_indices_from(K)] += noise
        
        try:
            L = np.linalg.cholesky(K)
            # Compute alpha = K^-1 y
            v = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, v)
            
            # Compute diagonal of K^-1
            # K^-1 = L^-T L^-1
            # Let M = L^-1. Then K^-1 = M.T @ M.
            # diag(M.T @ M) = sum(M^2, axis=0)
            M = np.linalg.solve(L, np.eye(n))
            K_inv_diag = np.sum(M**2, axis=0)
            
            # LOO predictions
            # mu_loo_i = y_i - alpha_i / K_inv_diag_i
            mu_loo = y_norm - alpha / K_inv_diag
            
            mse = np.mean((y_norm - mu_loo)**2)
            return mse
        except:
            return np.inf

    # Search for best length scale
    # Range of length scales
    # Estimate typical distance
    dists = np.abs(X[:, None] - X[None, :])
    np.fill_diagonal(dists, np.inf)
    med_dist = np.median(dists)
    if med_dist == 0: med_dist = 1.0
    
    # Log-uniform grid search
    log_ells = np.linspace(np.log(med_dist/10), np.log(med_dist*10), 50)
    best_ell = med_dist
    best_mse = np.inf
    
    for log_ell in log_ells:
        ell = np.exp(log_ell)
        mse = loo_cv_error(ell)
        if mse < best_mse:
            best_mse = mse
            best_ell = ell
        if time.time() > deadline - 5:
            break

    # If CV failed or is too high, maybe try a very small ell (interpolation)
    # For noiseless data, interpolation (small ell) is often best.
    # Let's check the MSE for a small ell.
    small_ell = med_dist * 0.01
    small_mse = loo_cv_error(small_ell)
    if small_mse < best_mse:
        best_ell = small_ell
        best_mse = small_mse

    # Final fit with best_ell
    K = rbf_kernel(X, X, best_ell)
    noise = 1e-8
    K[np.diag_indices_from(K)] += noise
    
    try:
        L = np.linalg.cholesky(K)
        v = np.linalg.solve(L, y_norm)
        alpha = np.linalg.solve(L.T, v)
    except:
        # Fallback to ridge regression
        K[np.diag_indices_from(K)] += 1e-4
        try:
            alpha = np.linalg.solve(K, y_norm)
        except:
            # Last resort: constant
            alpha = np.zeros(n)
            return lambda X: np.full((X.shape[0], 1), y_mean)

    def predict(X_test):
        X_test = X_test.flatten()
        K_s = rbf_kernel(X_test, X, best_ell)
        y_pred_norm = K_s @ alpha
        y_pred = y_pred_norm * y_std + y_mean
        return y_pred.reshape(-1, 1)
        
    return predict
