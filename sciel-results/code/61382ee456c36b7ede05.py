import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

# EVOLVE-BLOCK-START

def _rbf_kernel(X1, X2, length_scale):
    """Compute RBF kernel matrix between X1 and X2."""
    X1 = X1[:, np.newaxis]
    X2 = X2[:, np.newaxis]
    sq_dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + np.sum(X2 ** 2, axis=1)[np.newaxis, :] - 2 * X1 @ X2
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-0.5 * sq_dists / (length_scale ** 2))

def _fit_rbf(X, y, length_scale, noise_var):
    """Fit RBF kernel regression with given hyperparameters."""
    n = len(X)
    K = _rbf_kernel(X, X, length_scale)
    K_reg = K + noise_var * np.eye(n)
    try:
        alpha = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    
    def predict(X_new):
        K_new = _rbf_kernel(X_new, X, length_scale)
        return K_new @ alpha
    return predict

def _nlml_score(X, y, length_scale, noise_var):
    """Compute negative log marginal likelihood for RBF kernel."""
    n = len(X)
    K = _rbf_kernel(X, X, length_scale)
    K_reg = K + noise_var * np.eye(n)
    try:
        L = np.linalg.cholesky(K_reg)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        mse = np.sum(y ** 2) - np.sum(alpha ** 2)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * (n * np.log(2 * np.pi) + log_det + mse / noise_var)
        return nlml
    except np.linalg.LinAlgError:
        return 1e10

def _optimize_rbf(X, y):
    """Optimize RBF hyperparameters using NLML."""
    n = len(X)
    x_range = np.max(X) - np.min(X)
    if x_range < 1e-10:
        x_range = 1.0
    
    # Initial guesses
    init_ls = x_range / 3.0
    init_noise = np.var(y) * 0.1 + 1e-6
    
    best_nlml = np.inf
    best_params = (init_ls, init_noise)
    
    def objective(params):
        ls, nv = np.exp(params)
        return _nlml_score(X, y, ls, nv)
    
    x0 = [np.log(init_ls), np.log(init_noise)]
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_nlml:
            best_nlml = res.fun
            best_params = (np.exp(res.x[0]), np.exp(res.x[1]))
    except Exception:
        pass
    
    # Also try a few grid points as backup
    for ls in [x_range * f for f in [0.1, 0.3, 0.5, 1.0, 2.0]]:
        for nv in [np.var(y) * f + 1e-8 for f in [0.01, 0.1, 1.0]]:
            nlml = _nlml_score(X, y, ls, nv)
            if nlml < best_nlml:
                best_nlml = nlml
                best_params = (ls, nv)
    
    return best_params

def fit(X_train, y_train):
    start_time = time.time()
    TIME_LIMIT = 300.0
    SAFETY_MARGIN = 15.0
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    n, d = X_train.shape
    u_train = X_train[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(u_train)
    u_sorted = u_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    best_predict = None
    best_cv_mse = np.inf
    
    # Candidate 1: RBF kernel regression with NLML optimization
    try:
        if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
            ls, noise_var = _optimize_rbf(u_train, y_train)
            predict_rbf = _fit_rbf(u_train, y_train, ls, noise_var)
            
            # CV evaluation
            rng = np.random.RandomState(42)
            indices = np.arange(n)
            rng.shuffle(indices)
            folds = np.array_split(indices, 5)
            
            cv_mses = []
            for fold_idx in range(5):
                test_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(5) if i != fold_idx])
                
                u_tr = u_train[train_idx]
                y_tr = y_train[train_idx]
                u_te = u_train[test_idx]
                y_te = y_train[test_idx]
                
                # Quick fit with fixed hyperparameters (use the optimized ones)
                pred_tr = _fit_rbf(u_tr, y_tr, ls, noise_var)
                y_pred = pred_tr(u_te)
                if np.all(np.isfinite(y_pred)):
                    cv_mses.append(np.mean((y_pred - y_te) ** 2))
                else:
                    cv_mses.append(np.inf)
            
            mean_cv = np.mean(cv_mses)
            if mean_cv < best_cv_mse:
                best_cv_mse = mean_cv
                best_predict = predict_rbf
    except Exception:
        pass
    
    # Candidate 2: Spline interpolation with different smoothing
    for s in [0, 1, 10, 100, 1000]:
        try:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            spl = UnivariateSpline(u_sorted, y_sorted, s=s, k=3)
            
            def make_spline_pred(spl):
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    return spl(u)
                return predict
            
            predict_spline = make_spline_pred(spl)
            
            # CV evaluation
            rng = np.random.RandomState(42)
            indices = np.arange(n)
            rng.shuffle(indices)
            folds = np.array_split(indices, 5)
            
            cv_mses = []
            for fold_idx in range(5):
                test_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(5) if i != fold_idx])
                
                u_tr = u_train[train_idx]
                y_tr = y_train[train_idx]
                u_te = u_train[test_idx]
                y_te = y_train[test_idx]
                
                sort_tr = np.argsort(u_tr)
                spl_tr = UnivariateSpline(u_tr[sort_tr], y_tr[sort_tr], s=s, k=3)
                y_pred = spl_tr(u_te)
                if np.all(np.isfinite(y_pred)):
                    cv_mses.append(np.mean((y_pred - y_te) ** 2))
                else:
                    cv_mses.append(np.inf)
            
            mean_cv = np.mean(cv_mses)
            if mean_cv < best_cv_mse:
                best_cv_mse = mean_cv
                best_predict = predict_spline
        except Exception:
            pass
    
    # Candidate 3: Polynomial regression (degrees 1-6)
    for deg in range(1, 7):
        try:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            V = np.vander(u_train, N=deg + 1, increasing=True)
            lam = 1e-6
            A = V.T @ V + lam * np.eye(deg + 1)
            b = V.T @ y_train
            coeffs = np.linalg.solve(A, b)
            
            def make_poly_pred(coeffs):
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    V_test = np.vander(u, N=len(coeffs), increasing=True)
                    return V_test @ coeffs
                return predict
            
            predict_poly = make_poly_pred(coeffs)
            
            # CV evaluation
            rng = np.random.RandomState(42)
            indices = np.arange(n)
            rng.shuffle(indices)
            folds = np.array_split(indices, 5)
            
            cv_mses = []
            for fold_idx in range(5):
                test_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(5) if i != fold_idx])
                
                u_tr = u_train[train_idx]
                y_tr = y_train[train_idx]
                u_te = u_train[test_idx]
                y_te = y_train[test_idx]
                
                V_tr = np.vander(u_tr, N=deg + 1, increasing=True)
                A_tr = V_tr.T @ V_tr + lam * np.eye(deg + 1)
                b_tr = V_tr.T @ y_tr
                coeffs_tr = np.linalg.solve(A_tr, b_tr)
                V_te = np.vander(u_te, N=deg + 1, increasing=True)
                y_pred = V_te @ coeffs_tr
                if np.all(np.isfinite(y_pred)):
                    cv_mses.append(np.mean((y_pred - y_te) ** 2))
                else:
                    cv_mses.append(np.inf)
            
            mean_cv = np.mean(cv_mses)
            if mean_cv < best_cv_mse:
                best_cv_mse = mean_cv
                best_predict = predict_poly
        except Exception:
            pass
    
    # Fallback if nothing worked
    if best_predict is None:
        def predict_fallback(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            u = X[:, 0]
            # Simple linear interpolation
            return np.interp(u, u_sorted, y_sorted)
        best_predict = predict_fallback
    
    # Wrap final predict to ensure 2D input handling
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        u = X[:, 0]
        y_pred = best_predict(u)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        return y_pred
        
    return predict

# EVOLVE-BLOCK-END
