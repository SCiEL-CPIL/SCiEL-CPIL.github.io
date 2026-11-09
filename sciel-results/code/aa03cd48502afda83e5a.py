# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 315.0  # 15s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    # If y is constant, return constant
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Normalize X to [0, 1] for better kernel conditioning
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Center y for numerical stability
    y_centered = y_train - y_mean
    
    def check_time():
        return time.time() < deadline
    
    # GP Kernel: Matern + Linear + Constant
    # Params: [log(l_1), ..., log(l_d), log(sigma_f), log(sigma_n), log(b)]
    # l: lengthscales, sigma_f: signal variance, sigma_n: noise, b: linear coefficient
    
    def matern_kernel(X1, X2, l, nu=2.5):
        # X1: (n1, d), X2: (n2, d), l: (d,)
        # Compute squared distance
        # To avoid memory blowup for large n, we do chunking if needed, but n=200 is small.
        # dist^2 = sum_j (x1_j - x2_j)^2 / l_j^2
        # We can compute this efficiently.
        # X1_norm = X1 / l, X2_norm = X2 / l
        # dist2 = sum((X1_norm - X2_norm)^2)
        
        # For n=200, 200x200 matrix is fine.
        # X1: (n1, d), X2: (n2, d)
        # l: (d,)
        
        # Avoid division by zero
        l_safe = np.maximum(l, 1e-6)
        
        # Compute scaled coordinates
        # X1_s = X1 / l_safe
        # X2_s = X2 / l_safe
        # dist2 = sum(X1_s^2, axis=1)[:, None] + sum(X2_s^2, axis=1)[None, :] - 2 * X1_s @ X2_s.T
        
        X1_s = X1 / l_safe
        X2_s = X2 / l_safe
        
        sq1 = np.sum(X1_s ** 2, axis=1)[:, None]
        sq2 = np.sum(X2_s ** 2, axis=1)[None, :]
        cross = X1_s @ X2_s.T
        
        dist2 = sq1 + sq2 - 2 * cross
        # Clip small negative values due to floating point
        dist2 = np.maximum(dist2, 0.0)
        
        # Matern kernel
        # K = sigma_f^2 * (1 + sqrt(2*nu)*r + (2*nu/3)*r^2) * exp(-sqrt(2*nu)*r)
        # r = sqrt(dist2)
        r = np.sqrt(dist2)
        
        const1 = np.sqrt(2 * nu) * r
        const2 = (2 * nu / 3.0) * (r ** 2)
        
        K = np.exp(-const1) * (1 + const1 + const2)
        
        # Add linear kernel: b * (X1 @ X2.T)
        # Note: We use the original normalized X for linear part? Or scaled?
        # Usually linear kernel is on the original features. Let's use X1 @ X2.T.
        # But we need to be careful with scaling. Let's just use X1 @ X2.T.
        K_linear = b * (X1 @ X2.T)
        
        return K + K_linear
    
    def log_likelihood(params, X, y, is_training=True):
        # params: [log_l (d), log_sigma_f, log_sigma_n, log_b]
        log_l = params[:d]
        log_sigma_f = params[d]
        log_sigma_n = params[d+1]
        log_b = params[d+2]
        
        l = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        b = np.exp(log_b)
        
        K = matern_kernel(X, X, l)
        
        # Add signal variance
        K = K * (sigma_f ** 2)
        
        # Add noise
        K[np.diag_indices_from(K)] += sigma_n ** 2
        
        # Compute log likelihood
        # -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - n/2 * log(2*pi)
        try:
            # Cholesky decomposition for stability
            L = np.linalg.cholesky(K)
            # Solve L v = y
            v = np.linalg.solve(L, y)
            # log det K = 2 * sum(log(diag(L)))
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            # y^T K^-1 y = v^T v
            yKinvy = np.dot(v, v)
            
            ll = -0.5 * yKinvy - 0.5 * log_det_K - len(y) / 2.0 * np.log(2 * np.pi)
            
            if np.isnan(ll) or np.isinf(ll):
                return -np.inf
            return ll
            
        except np.linalg.LinAlgError:
            # Fallback to lstsq if cholesky fails
            try:
                Kinv_y = np.linalg.solve(K, y)
                log_det_K = np.linalg.slogdet(K)[1]
                yKinvy = np.dot(y, Kinv_y)
                ll = -0.5 * yKinvy - 0.5 * log_det_K - len(y) / 2.0 * np.log(2 * np.pi)
                if np.isnan(ll) or np.isinf(ll):
                    return -np.inf
                return ll
            except Exception:
                return -np.inf

    # 5-Fold CV
    def cv_score(params):
        if not check_time():
            return -np.inf
            
        indices = np.arange(n)
        np.random.seed(42) # Fixed seed for reproducibility in CV split
        np.random.shuffle(indices)
        
        fold_size = n // 5
        folds = np.array_split(indices, 5)
        
        total_ll = 0
        count = 0
        
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_tr = X_norm[tr_idx]
            y_tr = y_centered[tr_idx]
            X_val = X_norm[val_idx]
            y_val = y_centered[val_idx]
            
            ll = log_likelihood(params, X_tr, y_tr)
            if np.isinf(ll) or np.isnan(ll):
                return -np.inf
                
            total_ll += ll
            count += 1
            
            if not check_time():
                return -np.inf
                
        return total_ll / count

    # Initial parameters
    # Lengthscales: 0.5 (normalized space)
    # Signal variance: 1.0
    # Noise: 0.01
    # Linear coeff: 1.0
    init_params = np.zeros(d + 3)
    init_params[:d] = np.log(0.5)
    init_params[d] = np.log(1.0)
    init_params[d+1] = np.log(0.01)
    init_params[d+2] = np.log(1.0)
    
    best_params = init_params.copy()
    best_cv_score = cv_score(init_params)
    
    # Optimization
    # Bounds
    # log_l: [-5, 5] -> l in [0.006, 148]
    # log_sigma_f: [-5, 5]
    # log_sigma_n: [-10, 5]
    # log_b: [-5, 5]
    bounds = [(-5, 5)] * d + [(-5, 5), (-10, 5), (-5, 5)]
    
    # Try a few random restarts
    n_restarts = 3
    for r in range(n_restarts):
        if not check_time():
            break
            
        if r == 0:
            x0 = init_params.copy()
        else:
            x0 = init_params + np.random.randn(d + 3) * 0.5
            
        try:
            res = minimize(
                lambda p: -cv_score(p), 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 500}
            )
            
            if res.fun < -best_cv_score: # i.e. res.fun (negated) is smaller
                best_cv_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final prediction using all data with best params
    log_l = best_params[:d]
    log_sigma_f = best_params[d]
    log_sigma_n = best_params[d+1]
    log_b = best_params[d+2]
    
    l = np.exp(log_l)
    sigma_f = np.exp(log_sigma_f)
    sigma_n = np.exp(log_sigma_n)
    b = np.exp(log_b)
    
    K_train = matern_kernel(X_norm, X_norm, l)
    K_train = K_train * (sigma_f ** 2)
    K_train[np.diag_indices_from(K_train)] += sigma_n ** 2
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.solve(K_train, y_centered)
        except Exception:
            # Fallback to mean
            return lambda X: np.full(X.shape[0], y_mean)
    
    def predict(X):
        X_test = (X - X_min) / X_range
        K_test = matern_kernel(X_test, X_norm, l)
        K_test = K_test * (sigma_f ** 2)
        
        preds = K_test @ alpha + y_mean
        
        # Ensure finite
        preds = np.nan_to_num(preds, nan=y_mean, posinf=y_mean, neginf=y_mean)
        return preds

    return predict

# EVOLVE-BLOCK-END
