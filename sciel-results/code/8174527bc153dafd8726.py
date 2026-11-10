"""
Strategy: Gaussian Process Regression with Matérn-5/2 Kernel and Exact MLE.
This approach uses a non-parametric GP model where the kernel hyperparameters
(length scales per dimension, signal variance) are optimized via Maximum Likelihood
Estimation. This avoids the discrete grid search instability of KRR and adapts the
model's smoothness and scale to each specific hidden equation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 325.0  # Safety margin (limit is 330s)
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for stable optimization
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Pre-compute pairwise distances in normalized space
    # For n=200, this is a 200x200 matrix, very fast
    D = cdist(X_norm, X_norm, 'euclidean')
    
    # Matérn-5/2 Kernel
    # K(x, x') = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    # where r = sqrt(sum((x_i - x_i')^2 / l_i^2))
    # We use separate length scales per dimension (ARD)
    
    def matern52_kernel(D, log_ls, log_sigma_f):
        # log_ls: (d,)
        # log_sigma_f: scalar
        ls = np.exp(log_ls)
        sigma_f = np.exp(log_sigma_f)
        
        # Compute scaled distances
        # D is (n, n)
        # We need sqrt(sum(D_ij^2 / ls_i^2)) -> This is not directly possible with just Euclidean D
        # Wait, D is Euclidean distance. We need Mahalanobis distance with diagonal covariance.
        # It is better to compute the squared distances explicitly or use the property:
        # d^2 = sum((x_i - x_j)^2 / l_k^2)
        # Let's recompute the scaled distances properly.
        # X_norm is (n, d)
        # scaled_X = X_norm / ls
        # D_scaled = cdist(scaled_X, scaled_X, 'euclidean')
        
        scaled_X = X_norm / ls
        D_scaled = cdist(scaled_X, scaled_X, 'euclidean')
        
        r = D_scaled
        K = (1.0 + np.sqrt(5.0) * r + (5.0/3.0) * r**2) * np.exp(-np.sqrt(5.0) * r)
        K *= sigma_f**2
        return K
    
    def neg_log_likelihood(params):
        log_ls = params[:d]
        log_sigma_f = params[d]
        log_noise = params[d+1]
        
        K = matern52_kernel(D, log_ls, log_sigma_f)
        
        # Add small noise for numerical stability (noiseless data assumption)
        noise = np.exp(log_noise)
        K += noise * np.eye(n)
        
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
            
        # Solve L alpha = y
        alpha = np.linalg.solve(L, y_centered)
        
        # Log determinant: 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        
        # Negative Log Likelihood:
        # -0.5 * (y^T K^-1 y - log|K| + n*log(2*pi))
        # y^T K^-1 y = alpha^T y
        nll = 0.5 * (np.dot(alpha, y_centered) - log_det + n * np.log(2.0 * np.pi))
        
        return nll
    
    # Optimization
    # Initial guesses:
    # Length scales: 1.0 (log 0)
    # Signal variance: 1.0 (log 0)
    # Noise: 1e-8 (log ~ -18.4)
    
    init_ls = np.zeros(d)
    init_sigma = 0.0
    init_noise = np.log(1e-8)
    
    x0 = np.concatenate([init_ls, [init_sigma, init_noise]])
    
    # Bounds:
    # ls in [0.01, 10.0] -> log in [-4.6, 2.3]
    # sigma_f in [0.01, 100.0] -> log in [-4.6, 4.6]
    # noise in [1e-10, 1e-4] -> log in [-23, -9]
    
    bounds_ls = [(-4.6, 2.3)] * d
    bounds_sigma = [(-4.6, 4.6)]
    bounds_noise = [(-23.0, -9.0)]
    
    bounds = bounds_ls + bounds_sigma + bounds_noise
    
    best_params = x0
    best_nll = np.inf
    
    # Try a few restarts if time permits, or just one robust run
    # L-BFGS-B is robust. We will run one main optimization.
    
    try:
        res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4})
        
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # If we are running out of time, use current best
        pass
    else:
        # Maybe do a second pass with different init if first was bad?
        # For now, stick to one robust run to ensure stability.
        # If the first run failed (high NLL), try a default
        if best_nll > 1e5:
             best_params = x0
             best_nll = neg_log_likelihood(x0)
    
    # Extract best hyperparameters
    log_ls_best = best_params[:d]
    log_sigma_f_best = best_params[d]
    log_noise_best = best_params[d+1]
    
    # Final fit
    K = matern52_kernel(D, log_ls_best, log_sigma_f_best)
    noise = np.exp(log_noise_best)
    K += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y_centered)
    except np.linalg.LinAlgError:
        # Fallback to linear if Cholesky fails
        X_with_ones = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(X_with_ones, y_train, rcond=None)
        
        def predict_fallback(X):
            X_new_with_ones = np.column_stack([np.ones(X.shape[0]), X])
            return X_new_with_ones @ w
        
        return predict_fallback
    
    # Pre-compute L_inv_y for prediction? No, we need K(X_new, X_train) @ alpha
    # alpha is already computed.
    
    def predict(X):
        # Normalize new data
        X_new_norm = (X - min_vals) / ranges
        
        # Compute kernel between new and train
        ls = np.exp(log_ls_best)
        sigma_f = np.exp(log_sigma_f_best)
        
        scaled_X_new = X_new_norm / ls
        scaled_X_train = X_norm / ls
        
        D_new = cdist(scaled_X_new, scaled_X_train, 'euclidean')
        
        K_new = (1.0 + np.sqrt(5.0) * D_new + (5.0/3.0) * D_new**2) * np.exp(-np.sqrt(5.0) * D_new)
        K_new *= sigma_f**2
        
        y_pred = y_mean + K_new @ alpha
        return y_pred
    
    # Verify
    pred_train = predict(X_train)
    if not np.all(np.isfinite(pred_train)):
        X_with_ones = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(X_with_ones, y_train, rcond=None)
        
        def predict_fallback(X):
            X_new_with_ones = np.column_stack([np.ones(X.shape[0]), X])
            return X_new_with_ones @ w
        return predict_fallback
        
    return predict

# EVOLVE-BLOCK-END
