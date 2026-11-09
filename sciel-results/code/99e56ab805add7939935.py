import numpy as np
from scipy.optimize import minimize
import time

def fit(X_train, y_train):
    X = X_train
    y = y_train
    N, D = X.shape
    
    # Normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    noise_var = 1e-10 * np.var(y_norm)
    
    def get_kernel(dist_sq, l, sf, type):
        if type == 'gaussian':
            return sf * np.exp(-0.5 * dist_sq / (l**2))
        else: # matern 5/2
            r = np.sqrt(dist_sq) / l
            return sf * (1 + np.sqrt(3)*r + 0.6*r**2) * np.exp(-np.sqrt(3)*r)

    def compute_dist_sq(X1, X2):
        X1_sq = np.sum(X1**2, axis=1)[:, None]
        X2_sq = np.sum(X2**2, axis=1)[None, :]
        dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        return np.maximum(dist_sq, 0)

    def log_likelihood(params, X_tr, y_tr, k_type):
        log_l, log_sf = params
        l = np.exp(log_l)
        sf = np.exp(log_sf)
        
        dist_sq = compute_dist_sq(X_tr, X_tr)
        K = get_kernel(dist_sq, l, sf, k_type)
        K += noise_var * np.eye(len(X_tr))
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
        log_det_K = 2 * np.sum(np.log(np.diag(L)))
        
        nlml = 0.5 * np.dot(y_tr, alpha) + 0.5 * log_det_K + 0.5 * len(X_tr) * np.log(2 * np.pi)
        return nlml

    def optimize_hyperparams(X_tr, y_tr, k_type, max_iter=100):
        best_nlml = np.inf
        best_params = None
        for _ in range(3):
            init_log_l = np.random.uniform(-2, 2)
            init_log_sf = np.random.uniform(-2, 2)
            res = minimize(log_likelihood, [init_log_l, init_log_sf], method='Nelder-Mead', 
                           args=(X_tr, y_tr, k_type),
                           options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_nlml:
                best_nlml = res.fun
                best_params = res.x
        return best_params

    # 5-Fold CV to select kernel
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N // 5
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(5)]
    
    cv_scores = {'gaussian': [], 'matern': []}
    
    for k_type in ['gaussian', 'matern']:
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.setdiff1d(indices, val_idx)
            
            X_tr = X_norm[tr_idx]
            y_tr = y_norm[tr_idx]
            X_val = X_norm[val_idx]
            y_val = y_norm[val_idx]
            
            params = optimize_hyperparams(X_tr, y_tr, k_type, max_iter=50)
            if params is None:
                cv_scores[k_type].append(1e10)
                continue
                
            log_l, log_sf = params
            l = np.exp(log_l)
            sf = np.exp(log_sf)
            
            dist_tr = compute_dist_sq(X_tr, X_tr)
            K_tr = get_kernel(dist_tr, l, sf, k_type) + noise_var * np.eye(len(X_tr))
            L = np.linalg.cholesky(K_tr)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            
            dist_val = compute_dist_sq(X_val, X_tr)
            K_val = get_kernel(dist_val, l, sf, k_type)
            
            pred = K_val @ alpha
            mse = np.mean((pred - y_val)**2)
            cv_scores[k_type].append(mse)
            
    best_kernel = 'gaussian' if np.mean(cv_scores['gaussian']) < np.mean(cv_scores['matern']) else 'matern'
    
    # Refit on all data with best kernel
    final_params = optimize_hyperparams(X_norm, y_norm, best_kernel, max_iter=200)
    log_l, log_sf = final_params
    l = np.exp(log_l)
    sf = np.exp(log_sf)
    
    dist_tr = compute_dist_sq(X_norm, X_norm)
    K_train = get_kernel(dist_tr, l, sf, best_kernel) + noise_var * np.eye(N)
    L = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    
    def predict(X_test):
        X_t = X_test
        X_t_norm = (X_t - X_mean) / X_std
        dist_val = compute_dist_sq(X_t_norm, X_norm)
        K_test = get_kernel(dist_val, l, sf, best_kernel)
        pred_norm = K_test @ alpha
        pred = pred_norm * y_std + y_mean
        return pred

    return predict
