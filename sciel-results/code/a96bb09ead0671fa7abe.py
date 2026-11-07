import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin from 330s limit

    n_samples, n_features = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    if y_std < 1e-10:
        def predict_const(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], y_mean)
        return predict_const

    # Normalize y to help optimization
    y_norm = (y_train - y_mean) / y_std

    # Normalize X for better conditioning
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std

    def matern52_kernel(X1, X2, lengthscales, variance):
        # X1, X2 are already normalized
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * np.dot(X1_scaled, X2_scaled.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        dists = np.sqrt(sq_dists)
        
        sqrt5 = np.sqrt(5)
        K = variance * (1 + sqrt5 * dists + (5/3) * dists**2) * np.exp(-sqrt5 * dists)
        return K

    def log_marginal_likelihood(theta):
        if time.time() > deadline:
            return np.inf

        log_l = theta[:n_features]
        log_sigma_f = theta[n_features]
        log_sigma_n = theta[n_features + 1]
        
        lengthscales = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        K = matern52_kernel(X_norm, X_norm, lengthscales, sigma_f**2)
        K[np.diag_indices_from(K)] += sigma_n**2
        
        try:
            L, lower = cho_factor(K, lower=True, check_finite=False)
            alpha, _ = cho_solve((L, lower), y_norm, check_finite=False)
            logdet = 2 * np.sum(np.log(np.diag(lower)))
            
            n = n_samples
            lml = -0.5 * np.dot(y_norm, alpha) - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
            
            if not np.isfinite(lml):
                return np.inf
            return -lml
        except Exception:
            return np.inf

    # Initial parameters
    init_l = np.zeros(n_features)
    init_sigma_f = 0.0  # log(1.0)
    init_sigma_n = -2.0  # log(0.135)
    
    x0 = np.concatenate([init_l, [init_sigma_f, init_sigma_n]])
    
    bounds = [(-3, 3)] * (n_features + 2)
    
    best_lml = np.inf
    best_theta = x0
    
    # Optimization with multiple restarts
    rng = np.random.RandomState(42)
    
    # First try the initial point
    try:
        res = minimize(log_marginal_likelihood, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-8})
        if res.fun < best_lml:
            best_lml = res.fun
            best_theta = res.x
    except Exception:
        pass
    
    # Random restarts if time permits
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        rand_l = rng.uniform(-2, 2, size=n_features)
        rand_sf = rng.uniform(-2, 2)
        rand_sn = rng.uniform(-4, 0)
        x_rand = np.concatenate([rand_l, [rand_sf, rand_sn]])
        
        try:
            res_r = minimize(log_marginal_likelihood, x_rand, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-8})
            if res_r.fun < best_lml:
                best_lml = res_r.fun
                best_theta = res_r.x
        except Exception:
            pass

    # Final parameters
    final_log_l = best_theta[:n_features]
    final_log_sigma_f = best_theta[n_features]
    final_log_sigma_n = best_theta[n_features + 1]
    
    final_lengthscales = np.exp(final_log_l)
    final_sigma_f = np.exp(final_log_sigma_f)
    final_sigma_n = np.exp(final_log_sigma_n)
    
    # Pre-compute for prediction
    K_train = matern52_kernel(X_norm, X_norm, final_lengthscales, final_sigma_f**2)
    K_train[np.diag_indices_from(K_train)] += final_sigma_n**2
    
    try:
        L_train, lower_train = cho_factor(K_train, lower=True, check_finite=False)
        alpha, _ = cho_solve((L_train, lower_train), y_norm, check_finite=False)
    except Exception:
        # Fallback to simple linear regression if GP fails
        X_design = np.hstack([X_train, np.ones((n_samples, 1))])
        try:
            w, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
            def predict_lin(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                X_pred = np.hstack([X, np.ones((X.shape[0], 1))])
                return X_pred @ w
            return predict_lin
        except Exception:
            def predict_mean(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return np.full(X.shape[0], y_mean)
            return predict_mean

    # Capture for prediction
    f_X_mean = X_mean
    f_X_std = X_std
    f_lengthscales = final_lengthscales
    f_sigma_f = final_sigma_f
    f_alpha = alpha
    f_y_mean = y_mean
    f_y_std = y_std

    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm_test = (X - f_X_mean) / f_X_std
        K_test = matern52_kernel(X_norm_test, X_norm, f_lengthscales, f_sigma_f**2)
        
        mean_norm = K_test @ f_alpha
        mean_orig = mean_norm * f_y_std + f_y_mean
        
        return mean_orig

    return predict
