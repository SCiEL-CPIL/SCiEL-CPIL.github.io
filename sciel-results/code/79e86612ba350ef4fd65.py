import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm # Not needed for Matern, but good for RBF

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250.0 # 50s buffer for 4 equations + overhead
    
    X = X_train.flatten()
    y = y_train.flatten()
    n = len(X)
    
    # Normalize y for numerical stability in GP likelihood?
    # GP handles scale via sigma^2.
    
    # Kernel: Matern 5/2
    def matern52_kernel(X1, X2, length_scale, sigma):
        # X1, X2 are 1D arrays
        # Compute pairwise distances
        X1_col = X1[:, np.newaxis]
        X2_row = X2[np.newaxis, :]
        dist = np.sqrt(np.maximum(0, (X1_col - X2_row)**2)) # Euclidean in 1D is abs diff
        r = dist / length_scale
        return (sigma**2) * (1.0 + np.sqrt(5.0)*r + (5.0/3.0)*r**2) * np.exp(-np.sqrt(5.0)*r)

    def log_marginal_likelihood(params):
        # params: [log_sigma, log_length_scale]
        log_sigma, log_ls = params
        sigma = np.exp(log_sigma)
        ls = np.exp(log_ls)
        
        K = matern52_kernel(X, X, ls, sigma)
        # Add noise to diagonal for stability
        # Since noise is 0, use a small epsilon
        noise_var = 1e-6 * np.var(y) + 1e-10
        K += noise_var * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10 # Bad parameters
            
        alpha = np.linalg.solve(L, np.linalg.solve(L.T, y))
        
        # Log likelihood
        # -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - n/2 * log(2*pi)
        det_K = 2.0 * np.sum(np.log(np.diag(L)))
        ll = -0.5 * np.dot(y, alpha) - 0.5 * det_K - 0.5 * n * np.log(2.0 * np.pi)
        
        return -ll # Minimize negative log likelihood

    # Initial guesses
    # Length scale: roughly the range of X / 5
    x_range = np.max(X) - np.min(X)
    if x_range == 0:
        x_range = 1.0
    init_ls = x_range / 5.0
    init_sigma = np.std(y) if np.std(y) > 0 else 1.0
    
    x0 = [np.log(init_sigma), np.log(init_ls)]
    
    best_params = x0
    best_ll = -1e10
    
    # Optimization
    try:
        # Bounds:
        # sigma: 1e-4 to 1e4
        # ls: 1e-4 * x_range to 1e2 * x_range
        bounds = [(np.log(1e-4), np.log(1e4)), (np.log(1e-4 * x_range), np.log(1e2 * x_range))]
        
        # Use L-BFGS-B
        res = minimize(log_marginal_likelihood, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
        
        if res.success and res.fun < best_ll:
            best_params = res.x
            best_ll = res.fun
            
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10.0:
        # Fallback to simple polynomial if GP failed or is slow
        # But GP is fast. This is just a safety net.
        pass

    # Extract best hyperparameters
    log_sigma, log_ls = best_params
    sigma = np.exp(log_sigma)
    ls = np.exp(log_ls)
    
    K = matern52_kernel(X, X, ls, sigma)
    noise_var = 1e-6 * np.var(y) + 1e-10
    K += noise_var * np.eye(n)
    
    # Pre-compute Cholesky for prediction
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, np.linalg.solve(L.T, y))
    except np.linalg.LinAlgError:
        # Fallback to Ridge Regression on Polynomial features
        def features(X):
            X = X.flatten()
            cols = [np.ones_like(X)]
            for p in range(1, 8):
                cols.append(X**p)
            # Add trig
            for k in range(1, 5):
                cols.append(np.sin(k*X))
                cols.append(np.cos(k*X))
            return np.column_stack(cols)
        
        Xf = features(X)
        # Ridge
        lam = 1e-2
        w = np.linalg.lstsq(Xf + lam*np.eye(len(X)), y, rcond=None)[0]
        
        def predict(X_pred):
            Xp = X_pred.flatten()
            return features(Xp) @ w
            
        return predict

    def predict(X_pred):
        if time.time() > deadline:
             # Emergency fallback: mean
             return np.mean(y) * np.ones(len(X_pred))
             
        Xp = X_pred.flatten()
        
        # Compute K(x_pred, X_train)
        K_s = matern52_kernel(Xp, X, ls, sigma)
        
        # Prediction mean: K_s @ K^-1 @ y = K_s @ alpha
        mu = K_s @ alpha
        
        # Ensure finite
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        
        return mu

    return predict
