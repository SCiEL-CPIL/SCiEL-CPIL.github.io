"""
Strategy: Global Gaussian Process Regression with ARD Kernel.
Rationale:
1. The data is 1D and noiseless (200 points).
2. The hidden equations are likely smooth (polynomial, trigonometric, rational, etc.).
3. A GP with a Matern 5/2 or RBF kernel can perfectly interpolate smooth functions
   if the length-scale and signal variance are optimized.
4. Unlike piecewise splines, a global GP does not suffer from discontinuities at bin
   edges and generalizes better to the test points if the function is smooth across
   the domain.
5. We optimize the hyperparameters (length_scale, signal_var) to maximize the log
   marginal likelihood. This is robust and standard for 1D regression.
"""
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    X = X_train[:, 0].reshape(-1, 1)
    y = y_train.reshape(-1)
    
    n = len(X)
    
    # Normalize inputs to [0, 1] for better numerical stability of kernels
    x_min = X.min()
    x_max = X.max()
    if x_max > x_min:
        X_norm = (X - x_min) / (x_max - x_min)
    else:
        X_norm = np.zeros_like(X)
        
    # Normalize outputs? No, GP works in raw space usually, 
    # but scaling y helps if magnitude is huge. 
    # However, the kernel parameters adapt. Let's keep raw y for simplicity 
    # unless variance is extreme.
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def matern_52_kernel(X1, X2, length_scale):
        # Matern 5/2 kernel: k(x, x') = (1 + sqrt(5)*d/sigma + 5/3*(d/sigma)^2) * exp(-sqrt(5)*d/sigma)
        # where d is Euclidean distance.
        # For 1D, d = |x - x'|
        d = np.sqrt(np.sum((X1 - X2) ** 2, axis=1))
        dist = d / length_scale
        return (1 + np.sqrt(5) * dist + 5/3 * dist**2) * np.exp(-np.sqrt(5) * dist)
    
    def log_marginal_likelihood(params):
        # params: [log(length_scale), log(signal_var)]
        try:
            log_ls, log_sv = params
            length_scale = np.exp(log_ls)
            signal_var = np.exp(log_sv)
            
            K = matern_52_kernel(X_norm, X_norm, length_scale)
            K += 1e-6 * np.eye(n)  # Small jitter for numerical stability
            
            # Add noise variance? Data is noiseless, so noise var is tiny.
            # We treat it as exact interpolation with a tiny regularization.
            alpha = np.linalg.solve(K, y_norm)
            
            # Log marginal likelihood: -0.5 * y^T alpha - 0.5 * log(det(K)) - n/2 * log(2*pi)
            try:
                sign, logdet = np.linalg.slogdet(K)
                if sign <= 0:
                    return -1e10
                lml = -0.5 * np.dot(y_norm, alpha) - 0.5 * logdet
                return lml
            except np.linalg.LinAlgError:
                return -1e10
        except Exception:
            return -1e10
    
    def neg_log_marginal_likelihood(params):
        return -log_marginal_likelihood(params)
    
    # Initial guesses
    # Length scale in normalized space [0, 1]. 
    # Try a few initial points to avoid local minima.
    best_params = None
    best_lml = -np.inf
    
    # Search over initial length scales
    init_ls_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    init_sv_values = [1.0, 10.0]
    
    for ls in init_ls_values:
        for sv in init_sv_values:
            if time.time() > deadline:
                break
            x0 = np.array([np.log(ls), np.log(sv)])
            try:
                res = minimize(
                    neg_log_marginal_likelihood, 
                    x0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100}
                )
                if res.success and res.fun < -best_lml:
                    best_lml = -res.fun
                    best_params = res.x
            except Exception:
                pass
    
    if best_params is None:
        # Fallback if optimization fails
        best_params = np.array([np.log(0.5), np.log(1.0)])
        
    length_scale = np.exp(best_params[0])
    signal_var = np.exp(best_params[1])
    
    # Final fit with best parameters
    K = matern_52_kernel(X_norm, X_norm, length_scale)
    K += 1e-6 * np.eye(n)
    
    try:
        alpha = np.linalg.solve(K, y_norm)
    except np.linalg.LinAlgError:
        # Fallback to linear regression if GP fails
        coeffs = np.polyfit(X_train[:, 0], y_train, 3)
        def predict(X):
            x = X[:, 0]
            return np.polyval(coeffs, x)
        return predict
    
    def predict(X):
        x_pred = X[:, 0].reshape(-1, 1)
        # Normalize prediction inputs
        if x_max > x_min:
            x_norm_pred = (x_pred - x_min) / (x_max - x_min)
        else:
            x_norm_pred = np.zeros_like(x_pred)
            
        K_star = matern_52_kernel(x_norm_pred, X_norm, length_scale)
        
        # Prediction mean in normalized space
        mean_norm = np.dot(K_star, alpha)
        
        # Denormalize
        mean = mean_norm * y_std + y_mean
        
        return mean.reshape(X.shape[0])
    
    return predict
