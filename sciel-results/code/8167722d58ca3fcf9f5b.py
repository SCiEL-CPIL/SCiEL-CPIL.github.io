"""Symbolic regression via Gaussian Process Regression with Matérn-5/2 kernel."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel conditioning
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Matérn-5/2 kernel:
    # k(x, x') = sigma_f^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    # where r = ||x - x'|| / l
    
    def matern52_kernel(X1, X2, length_scale, sigma_f):
        # Compute squared distances
        # Using broadcasting: (n1, 1, d) - (1, n2, d)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        # Avoid negative zeros or tiny negatives due to floating point
        sq_dist = np.maximum(sq_dist, 0.0)
        r = np.sqrt(sq_dist) / length_scale
        
        return sigma_f**2 * (1 + np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)
    
    def neg_log_likelihood(params):
        if time.time() > DEADLINE:
            return 1e10
        length_scale, log_sigma_f, log_noise = params
        
        # Constrain parameters
        length_scale = np.exp(length_scale)
        sigma_f = np.exp(log_sigma_f)
        noise = np.exp(log_noise)
        
        try:
            K = matern52_kernel(X_norm, X_norm, length_scale, sigma_f)
            K[np.diag_indices_from(K)] += noise
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            
            # Solve L alpha = y
            alpha = np.linalg.solve(L, y_norm)
            
            # Log likelihood
            # -0.5 * y^T K^-1 y - 0.5 * log|K| - n/2 * log(2*pi)
            # log|K| = 2 * sum(log(diag(L)))
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            mse_term = 0.5 * np.dot(y_norm, alpha)
            
            nll = mse_term + 0.5 * log_det_K + 0.5 * n * np.log(2 * np.pi)
            
            if not np.isfinite(nll):
                return 1e10
                
            return nll
            
        except Exception:
            return 1e10

    # Initial hyperparameters
    # length_scale ~ 0.5, sigma_f ~ 1.0, noise ~ 1e-6 (noiseless data)
    init_params = [np.log(0.5), np.log(1.0), np.log(1e-6)]
    
    # Bounds for log parameters
    # length_scale in [0.01, 10] -> log in [-4.6, 2.3]
    # sigma_f in [0.01, 10] -> log in [-4.6, 2.3]
    # noise in [1e-8, 1e-2] -> log in [-18.4, -4.6]
    bounds = [(-4.6, 2.3), (-4.6, 2.3), (-18.4, -4.6)]
    
    best_params = init_params
    best_nll = neg_log_likelihood(init_params)
    
    # Optimize
    try:
        result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1000, 'maxfun': 10000})
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Check if we need a fallback or if optimization failed
    if not np.isfinite(best_nll):
        # Fallback: simple linear regression
        def predict_fallback(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            cols = [np.ones(len(X_test))]
            for j in range(d):
                cols.append(X_test_norm[:, j])
            feat = np.column_stack(cols)
            w_ls, *_ = np.linalg.lstsq(np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)]), y, rcond=None)
            return feat @ w_ls
        return predict_fallback

    length_scale = np.exp(best_params[0])
    sigma_f = np.exp(best_params[1])
    noise = np.exp(best_params[2])
    
    # Precompute Cholesky for prediction
    K = matern52_kernel(X_norm, X_norm, length_scale, sigma_f)
    K[np.diag_indices_from(K)] += noise
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y_norm)
    except np.linalg.LinAlgError:
        # If Cholesky fails, add jitter
        K += 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y_norm)
        except np.linalg.LinAlgError:
            # Final fallback
            def predict_fallback(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                X_test_norm = (X_test - X_min) / X_range
                cols = [np.ones(len(X_test))]
                for j in range(d):
                    cols.append(X_test_norm[:, j])
                feat = np.column_stack(cols)
                w_ls, *_ = np.linalg.lstsq(np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)]), y, rcond=None)
                return feat @ w_ls
            return predict_fallback

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        # Compute kernel between test points and training points
        K_s = matern52_kernel(X_test_norm, X_norm, length_scale, sigma_f)
        
        # Predictions: K_s @ K^-1 @ y
        # K^-1 = L^-T L^-1
        # So K_s @ K^-1 @ y = K_s @ (L^-T (L^-1 y))
        # We already have alpha = L^-T (L^-1 y) ? No, alpha = K^-1 y = L^-T L^-1 y.
        # So pred_norm = K_s @ alpha
        
        pred_norm = K_s @ alpha
        
        # Unnormalize
        pred = pred_norm * y_std + y_mean
        
        return pred

    return predict
# EVOLVE-BLOCK-END
