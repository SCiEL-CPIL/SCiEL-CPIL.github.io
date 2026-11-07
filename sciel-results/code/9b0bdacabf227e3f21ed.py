"""Gaussian Process Regression with Matern 5/2 kernel for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 330.0 - 10.0  # safety margin of 10 seconds
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs and outputs for better kernel conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    ys = (y - y_mean) / y_std
    
    # Compute pairwise distances (squared)
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2*xi.xj
    Xs_sq = np.sum(Xs**2, axis=1)
    dist_sq = Xs_sq[:, None] + Xs_sq[None, :] - 2 * Xs @ Xs.T
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)
    
    def matern52_kernel(dist, length_scale):
        """Matern 5/2 kernel."""
        r = dist / length_scale
        return np.exp(-np.sqrt(3) * r) * (1 + np.sqrt(3) * r + (3 * r**2) / 5)
    
    def compute_log_likelihood(length_scale, noise_var):
        """Compute log marginal likelihood for given hyperparameters."""
        K = matern52_kernel(dist, length_scale)
        K += noise_var * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        ll = -0.5 * (ys @ alpha + log_det + n * np.log(2 * np.pi))
        return ll
    
    # Optimize hyperparameters
    # Start with reasonable initial values
    best_ll = -1e10
    best_ls = 1.0
    best_noise = 1e-6
    
    # Grid search over length scales and noise
    ls_candidates = np.logspace(-2, 2, 15)  # from 0.01 to 100
    noise_candidates = [1e-8, 1e-6, 1e-4, 1e-2]
    
    # First pass: coarse grid
    for ls in ls_candidates:
        for noise in noise_candidates:
            if time.time() > deadline:
                break
            ll = compute_log_likelihood(ls, noise)
            if ll > best_ll:
                best_ll = ll
                best_ls = ls
                best_noise = noise
    
    # Second pass: fine grid around best
    if time.time() < deadline:
        ls_fine = np.linspace(best_ls * 0.5, best_ls * 2.0, 20)
        ls_fine = np.clip(ls_fine, 1e-3, 1000)
        noise_fine = np.array([best_noise * 0.5, best_noise, best_noise * 2.0])
        
        for ls in ls_fine:
            for noise in noise_fine:
                if time.time() > deadline:
                    break
                ll = compute_log_likelihood(ls, noise)
                if ll > best_ll:
                    best_ll = ll
                    best_ls = ls
                    best_noise = noise
    
    # Final optimization using L-BFGS-B
    if time.time() < deadline:
        from scipy.optimize import minimize
        
        def neg_ll(params):
            ls = np.exp(params[0])  # ensure positive
            noise = np.exp(params[1])
            return -compute_log_likelihood(ls, noise)
        
        x0 = np.array([np.log(best_ls), np.log(best_noise)])
        try:
            result = minimize(neg_ll, x0, method='L-BFGS-B', 
                             options={'maxiter': 50, 'maxfun': 200})
            if result.success or result.fun < -best_ll:
                best_ls = np.exp(result.x[0])
                best_noise = np.exp(result.x[1])
                best_ll = -result.fun
        except Exception:
            pass
    
    # Build the final GP model
    K = matern52_kernel(dist, best_ls)
    K += best_noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Add jitter if Cholesky fails
        K += 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback to least squares polynomial
            def features(X):
                cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + \
                       [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
                return np.column_stack(cols)
            w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
            def predict_fallback(X):
                X = np.asarray(X, dtype=np.float64)
                return features(X) @ w
            return predict_fallback
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        Xs_pred = (X - X_mean) / X_std
        
        # Compute kernel between test points and training points
        Xs_pred_sq = np.sum(Xs_pred**2, axis=1)
        dist_sq_pred = Xs_pred_sq[:, None] + Xs_sq[None, :] - 2 * Xs_pred @ Xs.T
        dist_sq_pred = np.maximum(dist_sq_pred, 0.0)
        dist_pred = np.sqrt(dist_sq_pred)
        
        K_star = matern52_kernel(dist_pred, best_ls)
        
        y_pred_std = K_star @ alpha
        y_pred = y_pred_std * y_std + y_mean
        
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
