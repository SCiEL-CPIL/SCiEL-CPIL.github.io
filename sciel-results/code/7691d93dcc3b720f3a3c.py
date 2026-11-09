"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def fit(X_train, y_train):
    X = X_train[:, 0]
    y = y_train
    n = len(X)
    
    # Normalize inputs to help with hyperparameter optimization
    X_mean = np.mean(X)
    X_std = np.std(X)
    if X_std == 0:
        X_std = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Standardize y for numerical stability in log-likelihood
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def matern52_kernel(x1, x2, length_scale, signal_variance):
        # x1: (n,), x2: (m,)
        # Returns (n, m)
        x1 = x1[:, np.newaxis]
        x2 = x2[np.newaxis, :]
        dist = np.sqrt((x1 - x2) ** 2)
        u = dist / length_scale
        k = signal_variance * (1 + np.sqrt(5) * u + 5/3 * u**2) * np.exp(-np.sqrt(5) * u)
        return k
    
    def log_marginal_likelihood(params):
        log_length_scale, log_signal_variance, log_noise_variance = params
        length_scale = np.exp(log_length_scale)
        signal_variance = np.exp(log_signal_variance)
        noise_variance = np.exp(log_noise_variance)
        
        try:
            K = matern52_kernel(X_norm, X_norm, length_scale, signal_variance)
            K[np.diag_indices_from(K)] += noise_variance
            
            # Solve K * alpha = y_norm
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            # Log likelihood: -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - n/2 * log(2*pi)
            # det(K) = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            
            ll = -0.5 * y_norm.dot(alpha) - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)
            
            # Add a small penalty for very small length scales or huge variances to prevent numerical issues
            # This acts as a prior
            if length_scale < 1e-3 or signal_variance > 1e6 or noise_variance > 1e6:
                return -1e6
            
            return -ll # We want to minimize negative log-likelihood
        
        except np.linalg.LinAlgError:
            return 1e6
    
    # Initial guesses
    # Length scale around 1.0, Signal variance around 1.0, Noise variance small
    initial_params = [0.0, 0.0, -5.0]
    
    # Optimize hyperparameters
    res = minimize(log_marginal_likelihood, initial_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.success:
        opt_params = res.x
    else:
        # Fallback to reasonable defaults if optimization fails
        opt_params = initial_params
    
    log_length_scale, log_signal_variance, log_noise_variance = opt_params
    length_scale = np.exp(log_length_scale)
    signal_variance = np.exp(log_signal_variance)
    noise_variance = np.exp(log_noise_variance)
    
    # Pre-compute Cholesky factor and alpha for training data
    K_train = matern52_kernel(X_norm, X_norm, length_scale, signal_variance)
    K_train[np.diag_indices_from(K_train)] += noise_variance
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        # If Cholesky fails, fall back to simple interpolation
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        def predict(X_new):
            x_new = X_new[:, 0]
            return np.interp(x_new, X_sorted, y_sorted)
        
        return predict
    
    def predict(X_new):
        x_new = X_new[:, 0]
        x_new_norm = (x_new - X_mean) / X_std
        
        # Compute kernel between test and train
        K_test_train = matern52_kernel(x_new_norm, X_norm, length_scale, signal_variance)
        
        # Prediction mean in normalized space
        mu_norm = K_test_train.dot(alpha)
        
        # Unnormalize
        mu = mu_norm * y_std + y_mean
        
        return mu

    return predict

# EVOLVE-BLOCK-END
