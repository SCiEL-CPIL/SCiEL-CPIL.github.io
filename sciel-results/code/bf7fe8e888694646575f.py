import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0 # seconds
    
    X = X_train.flatten()
    y = y_train.flatten()
    n = len(X)
    
    # Handle trivial case
    if np.std(y) < 1e-10:
        mean_y = np.mean(y)
        return lambda X_test: np.full(X_test.shape[0], mean_y)

    # Sort by X for consistency
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    candidates = []
    
    # 1. Polynomial Candidates
    # Degrees 3 to 10
    for deg in [3, 5, 7, 9, 11]:
        # Check time
        if time.time() > deadline - 30:
            break
        try:
            # Use np.polyfit for simplicity, but check condition number
            # Or use lstsq with Vandermonde
            V = np.vander(X_sorted, N=deg+1, increasing=True)
            # Center and scale X for numerical stability? 
            # np.polyfit does this internally, but let's use lstsq on scaled features
            x_mean = np.mean(X_sorted)
            x_std = np.std(X_sorted)
            if x_std < 1e-10:
                continue
            X_scaled = (X_sorted - x_mean) / x_std
            V_scaled = np.vander(X_scaled, N=deg+1, increasing=True)
            w, *_ = np.linalg.lstsq(V_scaled, y_sorted, rcond=None)
            
            # CV: 10-fold
            k = 10
            indices = np.arange(n)
            np.random.shuffle(indices)
            folds = np.array_split(indices, k)
            mses = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.setdiff1d(indices, val_idx)
                
                X_tr = X_sorted[train_idx]
                y_tr = y_sorted[train_idx]
                X_val = X_sorted[val_idx]
                y_val = y_sorted[val_idx]
                
                X_tr_scaled = (X_tr - x_mean) / x_std
                V_tr = np.vander(X_tr_scaled, N=deg+1, increasing=True)
                w_cv, *_ = np.linalg.lstsq(V_tr, y_tr, rcond=None)
                
                X_val_scaled = (X_val - x_mean) / x_std
                V_val = np.vander(X_val_scaled, N=deg+1, increasing=True)
                preds = V_val @ w_cv
                mse = np.mean((preds - y_val)**2)
                mses.append(mse)
            
            mean_mse = np.mean(mses)
            candidates.append({
                'type': 'poly',
                'deg': deg,
                'mse': mean_mse,
                'params': {'w': w, 'x_mean': x_mean, 'x_std': x_std}
            })
        except Exception:
            continue
            
    # 2. Spline Candidate
    if time.time() <= deadline - 30:
        try:
            cs = CubicSpline(X_sorted, y_sorted)
            # CV
            k = 10
            indices = np.arange(n)
            np.random.shuffle(indices)
            folds = np.array_split(indices, k)
            mses = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.setdiff1d(indices, val_idx)
                
                X_tr = X_sorted[train_idx]
                y_tr = y_sorted[train_idx]
                X_val = X_sorted[val_idx]
                y_val = y_sorted[val_idx]
                
                # Need unique X for spline
                if len(np.unique(X_tr)) < 4:
                    continue
                cs_cv = CubicSpline(X_tr, y_tr)
                preds = cs_cv(X_val)
                mse = np.mean((preds - y_val)**2)
                mses.append(mse)
            
            if mses:
                mean_mse = np.mean(mses)
                candidates.append({
                    'type': 'spline',
                    'mse': mean_mse,
                    'params': {'cs': cs}
                })
        except Exception:
            pass

    # 3. Gaussian Process Candidates
    if time.time() <= deadline - 60:
        # Matern 5/2 kernel
        def matern52(X1, X2, length_scale):
            r = np.sqrt(np.sum((X1[:, np.newaxis] - X2[np.newaxis, :])**2, axis=2))
            sr = r / length_scale
            return np.exp(-sr) * (1 + sr + (2/3)*sr**2)

        sigma_f_sq = np.var(y_sorted)
        sigma_n_sq = 1e-8
        
        length_scales = [0.1, 0.3, 0.7, 1.5, 3.0]
        
        for ls in length_scales:
            if time.time() > deadline - 30:
                break
            try:
                # CV
                k = 10
                indices = np.arange(n)
                np.random.shuffle(indices)
                folds = np.array_split(indices, k)
                mses = []
                for i in range(k):
                    val_idx = folds[i]
                    train_idx = np.setdiff1d(indices, val_idx)
                    
                    X_tr = X_sorted[train_idx]
                    y_tr = y_sorted[train_idx]
                    X_val = X_sorted[val_idx]
                    y_val = y_sorted[val_idx]
                    
                    K_train = matern52(X_tr, X_tr, ls) + sigma_n_sq * np.eye(len(X_tr))
                    K_cross = matern52(X_val, X_tr, ls)
                    
                    # Solve for alpha
                    try:
                        L = np.linalg.cholesky(K_train)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                        preds = K_cross @ alpha
                        mse = np.mean((preds - y_val)**2)
                        mses.append(mse)
                    except np.linalg.LinAlgError:
                        # Fallback to lstsq if Cholesky fails
                        try:
                            alpha = np.linalg.lstsq(K_train, y_tr, rcond=None)[0]
                            preds = K_cross @ alpha
                            mse = np.mean((preds - y_val)**2)
                            mses.append(mse)
                        except Exception:
                            continue
                
                if mses:
                    mean_mse = np.mean(mses)
                    candidates.append({
                        'type': 'gp',
                        'ls': ls,
                        'mse': mean_mse,
                        'params': {'sigma_f_sq': sigma_f_sq, 'sigma_n_sq': sigma_n_sq}
                    })
            except Exception:
                continue

    if not candidates:
        # Fallback to linear regression
        X_fit = np.column_stack([np.ones(n), X_sorted])
        w, *_ = np.linalg.lstsq(X_fit, y_sorted, rcond=None)
        return lambda X_test: np.column_stack([np.ones(X_test.shape[0]), X_test.flatten()]) @ w

    # Select best
    best = min(candidates, key=lambda x: x['mse'])
    
    # Fit best model on full data
    if best['type'] == 'poly':
        deg = best['deg']
        x_mean = best['params']['x_mean']
        x_std = best['params']['x_std']
        X_scaled = (X_sorted - x_mean) / x_std
        V = np.vander(X_scaled, N=deg+1, increasing=True)
        w, *_ = np.linalg.lstsq(V, y_sorted, rcond=None)
        
        def predict_poly(X_test):
            X_t = X_test.flatten()
            X_t_scaled = (X_t - x_mean) / x_std
            V_t = np.vander(X_t_scaled, N=deg+1, increasing=True)
            return V_t @ w
            
        return predict_poly
        
    elif best['type'] == 'spline':
        cs = best['params']['cs']
        def predict_spline(X_test):
            return cs(X_test.flatten())
        return predict_spline
        
    elif best['type'] == 'gp':
        ls = best['ls']
        sigma_f_sq = best['params']['sigma_f_sq']
        sigma_n_sq = best['params']['sigma_n_sq']
        
        K_train = matern52(X_sorted, X_sorted, ls) + sigma_n_sq * np.eye(n)
        try:
            L = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_sorted))
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K_train, y_sorted, rcond=None)[0]
            
        def predict_gp(X_test):
            X_t = X_test.flatten()
            K_cross = matern52(X_t, X_sorted, ls)
            return K_cross @ alpha
            
        return predict_gp

    # Should not reach here
    return lambda X_test: np.full(X_test.shape[0], np.mean(y_sorted))
