"""Symbolic regression via Gaussian Process regression with kernel search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0  # safety margin before 300s kill
    
    n_samples, n_features = X_train.shape
    
    # Standardize inputs for kernel computation
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    # Standardize outputs
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_std[y_std < 1e-8] = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    def rbf_kernel(X1, X2, length_scales, variance):
        """Compute RBF kernel with independent length scales."""
        # X1: (n1, d), X2: (n2, d)
        # length_scales: (d,)
        scaled_X1 = X1 / length_scales
        scaled_X2 = X2 / length_scales
        sq_dists = np.sum(scaled_X1**2, axis=1)[:, None] + np.sum(scaled_X2**2, axis=1)[None, :] - 2 * scaled_X1 @ scaled_X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return variance * np.exp(-0.5 * sq_dists)
    
    def compute_kernel_matrix(X, length_scales, variance):
        return rbf_kernel(X, X, length_scales, variance)
    
    def gp_predict(X_test, X_train_norm, y_train_norm, length_scales, variance, noise):
        K_train = compute_kernel_matrix(X_train_norm, length_scales, variance)
        K_test = rbf_kernel(X_test, X_train_norm, length_scales, variance)
        K_train += noise * np.eye(len(X_train_norm))
        
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-6 * np.eye(len(K_train))
            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                try:
                    alpha = np.linalg.solve(K_train, y_train_norm)
                except:
                    alpha = np.linalg.lstsq(K_train, y_train_norm, rcond=None)[0]
                pred = K_test @ alpha
                return pred * y_std + y_mean
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train_norm))
        pred = K_test @ alpha
        return pred * y_std + y_mean
    
    def gp_predict_with_cov(X_test, X_train_norm, y_train_norm, length_scales, variance, noise):
        """Return mean and variance of predictions."""
        K_train = compute_kernel_matrix(X_train_norm, length_scales, variance)
        K_test = rbf_kernel(X_test, X_train_norm, length_scales, variance)
        K_train += noise * np.eye(len(X_train_norm))
        
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-6 * np.eye(len(K_train))
            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                try:
                    alpha = np.linalg.solve(K_train, y_train_norm)
                except:
                    alpha = np.linalg.lstsq(K_train, y_train_norm, rcond=None)[0]
                pred = K_test @ alpha
                return pred * y_std + y_mean, np.zeros(len(X_test))
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train_norm))
        pred = K_test @ alpha
        var = variance - np.sum(K_test @ L, axis=1) ** 2 / np.sum(np.diag(L.T), axis=0)
        # Simpler variance computation
        v = np.linalg.solve(L, K_test.T)
        var = variance - np.sum(v**2, axis=0)
        var = np.maximum(var, 0)
        return pred * y_std + y_mean, var
    
    def neg_log_likelihood(params):
        """Negative log marginal likelihood for GP hyperparameters."""
        log_length_scales = params[:-2]
        log_variance = params[-2]
        log_noise = params[-1]
        
        length_scales = np.exp(log_length_scales)
        variance = np.exp(log_variance)
        noise = np.exp(log_noise)
        
        K = compute_kernel_matrix(X_norm, length_scales, variance) + noise * np.eye(n_samples)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e6
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        nll = 0.5 * y_norm @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n_samples * np.log(2 * np.pi)
        return nll
    
    # Initial hyperparameters
    # Use median distance as initial length scale
    if n_samples > 1:
        dists = cdist(X_norm, X_norm, 'euclidean')
        median_dist = np.median(dists[np.triu_indices(n_samples, k=1)])
        if median_dist < 1e-8:
            median_dist = 1.0
        init_ls = np.full(n_features, median_dist)
    else:
        init_ls = np.ones(n_features)
    
    init_log_ls = np.log(init_ls)
    init_log_var = np.log(y_norm.var() + 1e-8)
    init_log_noise = np.log(1e-4)
    
    init_params = np.concatenate([init_log_ls, [init_log_var, init_log_noise]])
    
    # Try multiple initializations and find best GP
    best_nll = np.inf
    best_params = init_params
    best_pred_train = None
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params = init_params.copy()
        else:
            # Random perturbation
            params = init_params.copy()
            params[:-2] += np.random.randn(n_features) * 0.5
            params[-2] += np.random.randn() * 0.5
            params[-1] += np.random.randn() * 0.5
        
        try:
            result = minimize(neg_log_likelihood, params, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-6})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Extract best hyperparameters
    log_length_scales = best_params[:-2]
    log_variance = best_params[-2]
    log_noise = best_params[-1]
    
    length_scales = np.exp(log_length_scales)
    variance = np.exp(log_variance)
    noise = np.exp(log_noise)
    
    # Compute final GP model
    K_train = compute_kernel_matrix(X_norm, length_scales, variance) + noise * np.eye(n_samples)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        K_train += 1e-6 * np.eye(n_samples)
        try:
            L = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K_train, y_norm, rcond=None)[0]
    
    # Check if we have time for a second kernel (e.g., polynomial)
    elapsed = time.time() - start_time
    use_poly_kernel = elapsed < 100 and n_features <= 3
    
    if use_poly_kernel:
        # Add polynomial kernel component
        def poly_kernel(X1, X2, degree=2, coef0=1.0):
            return (coef0 + X1 @ X2.T) ** degree
        
        # Fit a combined kernel: RBF + Poly
        # For simplicity, use a weighted combination
        # We'll try a few weights
        
        best_combined_nll = np.inf
        best_weight = 0.5
        
        # Precompute poly kernel
        K_poly = poly_kernel(X_norm, X_norm, degree=2, coef0=1.0)
        K_rbf = compute_kernel_matrix(X_norm, length_scales, variance)
        
        for weight in [0.0, 0.2, 0.5, 0.8, 1.0]:
            if time.time() > deadline - 20:
                break
            K_combined = weight * K_rbf + (1 - weight) * K_poly + noise * np.eye(n_samples)
            try:
                L_c = np.linalg.cholesky(K_combined)
                alpha_c = np.linalg.solve(L_c.T, np.linalg.solve(L_c, y_norm))
                nll_c = 0.5 * y_norm @ alpha_c + np.sum(np.log(np.diag(L_c))) + 0.5 * n_samples * np.log(2 * np.pi)
                if nll_c < best_combined_nll:
                    best_combined_nll = nll_c
                    best_weight = weight
            except:
                continue
        
        if best_weight < 1.0:
            K_combined = best_weight * K_rbf + (1 - best_weight) * K_poly + noise * np.eye(n_samples)
            try:
                L = np.linalg.cholesky(K_combined)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            except:
                alpha = np.linalg.lstsq(K_combined, y_norm, rcond=None)[0]
            
            def predict_combined(X):
                X_test_norm = (X - X_mean) / X_std
                K_test_rbf = rbf_kernel(X_test_norm, X_norm, length_scales, variance)
                K_test_poly = poly_kernel(X_test_norm, X_norm, degree=2, coef0=1.0)
                K_test = best_weight * K_test_rbf + (1 - best_weight) * K_test_poly
                pred = K_test @ alpha
                return pred * y_std + y_mean
        else:
            def predict_combined(X):
                X_test_norm = (X - X_mean) / X_std
                K_test = rbf_kernel(X_test_norm, X_norm, length_scales, variance)
                pred = K_test @ alpha
                return pred * y_std + y_mean
    else:
        def predict_combined(X):
            X_test_norm = (X - X_mean) / X_std
            K_test = rbf_kernel(X_test_norm, X_norm, length_scales, variance)
            pred = K_test @ alpha
            return pred * y_std + y_mean
    
    # Final prediction function
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = predict_combined(X)
        # Ensure finite
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds
    
    return predict
