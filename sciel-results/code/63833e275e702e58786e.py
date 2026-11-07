"""Symbolic regression with ensemble of Kernel Ridge Regressors and adaptive features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global time tracker
_total_start = None
_time_limit = 280.0  # Safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _rbf_kernel(X1, X2, length_scales=None, sigma_f=1.0):
    """Compute RBF kernel between X1 and X2 with optional ARD length scales."""
    if length_scales is not None:
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
    else:
        X1_scaled = X1
        X2_scaled = X2
    
    sq_norms1 = np.sum(X1_scaled ** 2, axis=1)
    sq_norms2 = np.sum(X2_scaled ** 2, axis=1)
    sq_dists = sq_norms1[:, None] + sq_norms2[None, :] - 2 * X1_scaled @ X2_scaled.T
    np.maximum(sq_dists, 0, out=sq_dists)
    return sigma_f ** 2 * np.exp(-0.5 * sq_dists)

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Build rich feature set
    def build_features(X):
        n = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n)]
        
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        
        # Polynomial terms up to degree 3
        for deg in range(2, 4):
            def _gen(idx, remaining, current):
                if idx == d - 1:
                    yield current + [remaining]
                    return
                for k in range(remaining + 1):
                    yield from _gen(idx + 1, remaining - k, current + [k])
            
            for exps in _gen(0, deg, []):
                term = np.ones(n)
                for j in range(d):
                    if exps[j] > 0:
                        term = term * np.power(X[:, j], exps[j])
                cols.append(term)
        
        # Trigonometric terms
        for j in range(d):
            cols.append(np.sin(np.pi * X[:, j]))
            cols.append(np.cos(np.pi * X[:, j]))
        
        # Interactions
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(np.sin(np.pi * (X[:, i] + X[:, j])))
        
        # Exponential and log
        for j in range(d):
            cols.append(np.exp(X[:, j] / 3.0))
            cols.append(np.log(X[:, j] + 0.5))
        
        # Safe ratios
        for i in range(d):
            for j in range(i + 1, d):
                denom = X[:, j] + 0.5
                cols.append(X[:, i] / denom)
        
        return np.column_stack(cols)
    
    F = build_features(X)
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    n_features = F_norm.shape[1]
    
    # Strategy 1: Ensemble of Kernel Ridge Regressors with different kernels
    # We'll fit multiple KRR models and average their predictions
    
    krr_models = []
    
    # RBF kernel with different length scales
    rbf_configs = [
        {'length_scales': None, 'sigma_f': 1.0, 'alpha': 1e-4},
        {'length_scales': None, 'sigma_f': 1.0, 'alpha': 1e-3},
        {'length_scales': None, 'sigma_f': 1.0, 'alpha': 1e-2},
        {'length_scales': np.ones(d), 'sigma_f': 1.0, 'alpha': 1e-4},
        {'length_scales': np.ones(d) * 0.5, 'sigma_f': 1.0, 'alpha': 1e-4},
        {'length_scales': np.ones(d) * 2.0, 'sigma_f': 1.0, 'alpha': 1e-4},
    ]
    
    for cfg in rbf_configs:
        if _remaining() < 2.0:
            break
        
        K = _rbf_kernel(X_norm, X_norm, cfg['length_scales'], cfg['sigma_f'])
        alpha = cfg['alpha']
        
        try:
            w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
            pred = K @ w
            mse = np.mean((pred - y_norm) ** 2)
            
            if mse < 0.5:  # Only keep models that fit reasonably well
                krr_models.append({
                    'type': 'rbf',
                    'cfg': cfg,
                    'w': w,
                    'score': mse
                })
        except np.linalg.LinAlgError:
            pass
    
    # Polynomial kernel
    poly_configs = [
        {'degree': 2, 'gamma': 1.0, 'alpha': 1e-4},
        {'degree': 2, 'gamma': 2.0, 'alpha': 1e-4},
        {'degree': 3, 'gamma': 0.5, 'alpha': 1e-4},
    ]
    
    for cfg in poly_configs:
        if _remaining() < 2.0:
            break
        
        K = (cfg['gamma'] * X_norm @ X_norm.T + 1) ** cfg['degree']
        alpha = cfg['alpha']
        
        try:
            w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
            pred = K @ w
            mse = np.mean((pred - y_norm) ** 2)
            
            if mse < 0.5:
                krr_models.append({
                    'type': 'poly',
                    'cfg': cfg,
                    'w': w,
                    'score': mse
                })
        except np.linalg.LinAlgError:
            pass
    
    # Strategy 2: Ridge regression on rich features
    ridge_models = []
    lam_values = np.logspace(-3, 3, 7)
    
    for lam in lam_values:
        if _remaining() < 2.0:
            break
        
        A = F_norm.T @ F_norm + lam * np.eye(n_features)
        b = F_norm.T @ y_norm
        
        try:
            w = np.linalg.solve(A, b)
            pred = F_norm @ w
            mse = np.mean((pred - y_norm) ** 2)
            
            ridge_models.append({
                'w': w,
                'lam': lam,
                'score': mse
            })
        except np.linalg.LinAlgError:
            pass
    
    # Strategy 3: For 1D, try specific nonlinear forms
    nonlinear_model = None
    
    if d == 1:
        x = X[:, 0]
        best_nl_score = np.inf
        best_nl_type = None
        best_nl_params = None
        
        # Power law
        def obj_power(params):
            a, p, b = params
            try:
                pred = a * np.power(x, p) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_power, [1.0, 1.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300})
            if res.fun < best_nl_score:
                best_nl_score = res.fun
                best_nl_type = 'power'
                best_nl_params = res.x
        except:
            pass
        
        # Exponential
        def obj_exp(params):
            a, p, b = params
            try:
                pred = a * np.exp(p * x) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_exp, [1.0, 0.5, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300})
            if res.fun < best_nl_score:
                best_nl_score = res.fun
                best_nl_type = 'exp'
                best_nl_params = res.x
        except:
            pass
        
        # Sine
        def obj_sin(params):
            a, p, phi, b = params
            try:
                pred = a * np.sin(p * x + phi) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_sin, [1.0, 2.0, 0.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300})
            if res.fun < best_nl_score:
                best_nl_score = res.fun
                best_nl_type = 'sin'
                best_nl_params = res.x
        except:
            pass
        
        if best_nl_type is not None and best_nl_score < 0.5 * y_std ** 2:
            nonlinear_model = {
                'type': best_nl_type,
                'params': best_nl_params,
                'score': best_nl_score / (y_std ** 2 + 1e-10)
            }
    
    # Build ensemble prediction
    # We'll use a weighted average of the best models from each strategy
    
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_min) / X_range
        
        preds = []
        weights = []
        
        # KRR predictions
        for model in krr_models:
            if model['type'] == 'rbf':
                K_test = _rbf_kernel(X_t_norm, X_norm, model['cfg']['length_scales'], model['cfg']['sigma_f'])
                y_pred_norm = K_test @ model['w']
            elif model['type'] == 'poly':
                K_test = (model['cfg']['gamma'] * X_t_norm @ X_norm.T + 1) ** model['cfg']['degree']
                y_pred_norm = K_test @ model['w']
            
            preds.append(y_pred_norm)
            weights.append(1.0 / (model['score'] + 1e-6))
        
        # Ridge predictions
        for model in ridge_models:
            F_t = build_features(X_t)
            F_t_norm = (F_t - F_mean) / F_std
            y_pred_norm = F_t_norm @ model['w']
            preds.append(y_pred_norm)
            weights.append(1.0 / (model['score'] + 1e-6))
        
        # Nonlinear prediction (1D only)
        if nonlinear_model is not None:
            x = X_t[:, 0]
            if nonlinear_model['type'] == 'power':
                a, p, b = nonlinear_model['params']
                y_pred = a * np.power(x, p) + b
            elif nonlinear_model['type'] == 'exp':
                a, p, b = nonlinear_model['params']
                y_pred = a * np.exp(p * x) + b
            elif nonlinear_model['type'] == 'sin':
                a, p, phi, b = nonlinear_model['params']
                y_pred = a * np.sin(p * x + phi) + b
            else:
                y_pred = np.full(X_t.shape[0], y_mean)
            
            preds.append((y_pred - y_mean) / y_std)
            weights.append(1.0 / (nonlinear_model['score'] + 1e-6))
        
        # If no models, fall back to mean
        if len(preds) == 0:
            return np.full(X_t.shape[0], y_mean)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        stacked_preds = np.column_stack(preds)
        y_pred_norm = stacked_preds @ weights
        
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        mask = ~np.isfinite(y_pred)
        if np.any(mask):
            y_pred = y_pred.copy()
            y_pred[mask] = y_mean
        
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
