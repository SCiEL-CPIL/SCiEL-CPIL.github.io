"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 300s limit
    
    X = X_train
    y = y_train
    d = X.shape[1]
    n = X.shape[0]
    
    # Normalize inputs and target for better GP conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Kernel: Matern 5/2 + White noise
    # Parameters: [length_scale_1, ..., length_scale_d, signal_var, noise_var]
    def matern52(X1, X2, lengths):
        # X1: (n, d), X2: (m, d)
        # lengths: (d,)
        # Compute scaled distance
        # r = sqrt(sum((x1 - x2)^2 / l^2))
        # We want to avoid division by zero or very small lengths
        lengths = np.maximum(lengths, 1e-8)
        X1_scaled = X1 / lengths
        X2_scaled = X2 / lengths
        
        # Efficient squared distance
        # ||a||^2 + ||b||^2 - 2a.b
        sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * np.dot(X1_scaled, X2_scaled.T)
        sq_dists = np.maximum(sq_dists, 0)
        r = np.sqrt(sq_dists)
        
        sqrt3_r = np.sqrt(3) * r
        K = (1 + sqrt3_r + (sqrt3_r**2) / 3) * np.exp(-sqrt3_r)
        return K

    def log_marginal_likelihood(neg_params):
        # neg_params is the negative of the actual parameters for maximization
        params = -neg_params
        lengths = params[:d]
        signal_var = params[d]
        noise_var = params[d+1]
        
        K = matern52(X_norm, X_norm, lengths)
        K += (noise_var + 1e-10) * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, alpha)
            lml = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e10

    # Initial guess: lengths=1, signal_var=1, noise_var=1e-3
    init_params = np.ones(d + 2)
    init_params[d+1] = 1e-3
    init_neg_params = -init_params
    
    # Optimize
    best_params = init_params
    best_lml = 1e10
    
    try:
        # Use L-BFGS-B
        res = minimize(log_marginal_likelihood, init_neg_params, method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-4})
        
        if time.time() < deadline:
            if -res.fun < best_lml:
                best_params = -res.x
                best_lml = -res.fun
    except Exception:
        pass
    
    # If optimization failed or didn't improve, use initial or fallback
    if best_lml >= 1e9:
        best_params = init_params

    lengths = best_params[:d]
    signal_var = best_params[d]
    noise_var = best_params[d+1]
    
    # Precompute Cholesky for training data for prediction
    K_train = matern52(X_norm, X_norm, lengths)
    K_train += (noise_var + 1e-10) * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L, y_norm)
        alpha = np.linalg.solve(L.T, alpha)
    except np.linalg.LinAlgError:
        # Fallback to ridge regression if GP fails
        K_reg = matern52(X_norm, X_norm, lengths) + 1e-2 * np.eye(n)
        try:
            alpha = np.linalg.solve(K_reg, y_norm)
            L = None
        except:
            # Final fallback: mean predictor
            def predict_fallback(X_test):
                return np.ones(X_test.shape[0]) * y_mean
            return predict_fallback

    def predict(X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test_norm = (X_test - X_mean) / X_std
        
        # Compute K(X_test, X_train)
        K_test_train = matern52(X_test_norm, X_norm, lengths)
        
        if L is not None:
            v = np.linalg.solve(L, K_test_train.T)
            y_pred_norm = K_test_train @ v.T @ alpha
        else:
            y_pred_norm = K_test_train @ alpha
            
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
        
        if y_pred.shape[0] == 1 and X_test.shape[0] == 1:
             return y_pred[0]
        return y_pred

    return predict
# EVOLVE-BLOCK-END
