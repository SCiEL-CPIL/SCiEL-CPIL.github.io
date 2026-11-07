import numpy as np
from scipy.optimize import minimize

def fit(X_train, y_train):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_std[y_std == 0] = 1.0
    
    X_std_train = (X_train - X_mean) / X_std
    y_std_train = (y_train - y_mean) / y_std
    
    n = X_std_train.shape[0]
    
    def rbf_kernel(X1, X2, length_scale):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-0.5 * sq_dists / (length_scale * length_scale))
    
    def log_marginal_likelihood(params):
        log_l = params[0]
        log_sigma_f = params[1]
        log_sigma_n = params[2]
        
        l = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        K = rbf_kernel(X_std_train, X_std_train, l) * (sigma_f * sigma_f)
        K[np.diag_indices(n)] += sigma_n * sigma_n
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e100
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_std_train))
        
        log_det = 2 * np.sum(np.log(np.diag(L)))
        ll = -0.5 * y_std_train.T @ alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        
        return -ll
    
    inits = [
        np.array([0.0, 0.0, -2.0]),
        np.array([-1.0, 0.0, -2.0]),
        np.array([1.0, 0.0, -2.0]),
        np.array([0.0, 1.0, -2.0]),
        np.array([0.0, -1.0, -2.0]),
        np.array([0.5, 0.5, -1.0]),
        np.array([-0.5, 0.5, -1.0]),
    ]
    
    best_loss = np.inf
    best_params = inits[0]
    
    for init in inits:
        try:
            res = minimize(log_marginal_likelihood, init, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-6})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            pass
            
    log_l, log_sigma_f, log_sigma_n = best_params
    l = np.exp(log_l)
    sigma_f = np.exp(log_sigma_f)
    sigma_n = np.exp(log_sigma_n)
    
    K_train = rbf_kernel(X_std_train, X_std_train, l) * (sigma_f * sigma_f)
    K_train[np.diag_indices(n)] += sigma_n * sigma_n
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_std_train))
    except np.linalg.LinAlgError:
        # Fallback to simple linear regression if GP fails
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
        
    def predict(X):
        X_std = (X - X_mean) / X_std
        K_test = rbf_kernel(X_std, X_std_train, l) * (sigma_f * sigma_f)
        y_pred_std = K_test @ alpha
        y_pred = y_pred_std * y_std + y_mean
        return y_pred
        
    return predict
