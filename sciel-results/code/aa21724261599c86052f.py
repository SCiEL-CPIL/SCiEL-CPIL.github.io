"""Symbolic regression via multi-strategy ensemble with hold-out validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import warnings

warnings.filterwarnings('ignore')


def _safe_predict(X, model):
    """Safely compute predictions, returning finite values."""
    try:
        preds = model(X)
        if not np.all(np.isfinite(preds)):
            # Replace non-finite with mean of finite
            finite = np.isfinite(preds)
            if np.any(finite):
                preds[~finite] = np.mean(preds[finite])
            else:
                preds = np.zeros_like(preds)
        return preds
    except Exception:
        return np.zeros(len(X))


def _nmse(preds, y):
    """Normalized MSE."""
    var_y = np.var(y)
    if var_y < 1e-15:
        return 0.0
    return np.mean((preds - y) ** 2) / var_y


def _fit_polynomial(X, y, degree):
    """Fit polynomial of given degree."""
    d = X.shape[1]
    # For 1D, use np.polyfit
    if d == 1:
        x = X.flatten()
        coeffs = np.polyfit(x, y, degree)
        return lambda X_: np.polyval(coeffs, X_.flatten())
    else:
        # Multivariate polynomial features
        from itertools import combinations_with_replacement
        terms = []
        for total_deg in range(degree + 1):
            for idx in combinations_with_replacement(range(d), total_deg):
                terms.append(idx)
        
        def make_features(X_):
            feats = []
            for idx in terms:
                col = np.ones(len(X_))
                for j in idx:
                    col = col * X_[:, j]
                feats.append(col)
            return np.column_stack(feats)
        
        F_train = make_features(X)
        w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
        return lambda X_: make_features(X_) @ w


def _fit_spline(X, y, s_factor):
    """Fit smoothing spline."""
    d = X.shape[1]
    if d == 1:
        x = X.flatten()
        # Sort by x for spline
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        
        # Determine s based on data variance
        var_y = np.var(y)
        s = s_factor * var_y * len(y)
        
        try:
            sp = UnivariateSpline(x_sorted, y_sorted, s=s)
            return lambda X_: sp(X_.flatten())
        except Exception:
            # Fallback to polynomial
            return _fit_polynomial(X, y, 5)
    else:
        # For multi-dim, fall back to polynomial
        return _fit_polynomial(X, y, 4)


def _fit_gaussian_process(X, y):
    """Fit a simple GP with RBF kernel using MLE."""
    d = X.shape[1]
    n = len(y)
    
    # Normalize
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-10
    y_mean = y.mean()
    y_std = y.std() + 1e-10
    
    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    
    def log_marginal(theta):
        log_l = theta[0]  # log length_scale
        log_alpha = theta[1]  # log signal variance
        log_noise = theta[2]  # log noise variance
        
        l = np.exp(log_l)
        alpha = np.exp(log_alpha)
        noise = np.exp(log_noise)
        
        # RBF kernel
        sq_dists = np.sum(X_norm ** 2, axis=1)
        K = alpha * np.exp(-0.5 * (np.outer(sq_dists, np.ones(n)) + np.outer(np.ones(n), sq_dists) - 2 * X_norm @ X_norm.T) / l ** 2)
        K[np.diag_indices_from(K)] += noise
        
        try:
            L = np.linalg.cholesky(K)
            alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            lml = -0.5 * y_norm @ alpha_vec - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e10
    
    # Initial guesses
    theta0 = [0.0, 0.0, -2.0]
    
    try:
        res = minimize(log_marginal, theta0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 0.1, 'fatol': 0.1})
        if res.success or res.fun < 1e10:
            l = np.exp(res.x[0])
            alpha = np.exp(res.x[1])
            noise = np.exp(res.x[2])
            
            sq_dists = np.sum(X_norm ** 2, axis=1)
            K = alpha * np.exp(-0.5 * (np.outer(sq_dists, np.ones(n)) + np.outer(np.ones(n), sq_dists) - 2 * X_norm @ X_norm.T) / l ** 2)
            K[np.diag_indices_from(K)] += noise
            
            L = np.linalg.cholesky(K)
            alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            def predict(X_):
                X_n = (X_ - x_mean) / x_std
                sq_d = np.sum(X_n ** 2, axis=1)
                sq_d_train = np.sum(X_norm ** 2, axis=1)
                K_star = alpha * np.exp(-0.5 * (np.outer(sq_d, np.ones(n)) + np.outer(np.ones(n), sq_d_train) - 2 * X_n @ X_norm.T) / l ** 2)
                pred_norm = K_star @ alpha_vec
                return pred_norm * y_std + y_mean
            
            return predict
        else:
            raise Exception("GP optimization failed")
    except Exception:
        # Fallback
        return _fit_polynomial(X, y, 4)


def _fit_rbf_network(X, y, n_centers=20):
    """Fit RBF network via least squares after selecting centers."""
    d = X.shape[1]
    n = len(y)
    
    # Select centers via random sampling
    rng = np.random.RandomState(42)
    center_idx = rng.choice(n, min(n_centers, n), replace=False)
    centers = X[center_idx]
    
    # Estimate width from median distance
    if d == 1:
        x = X.flatten()
        c = centers.flatten()
        dists = np.abs(x[:, None] - c[None, :])
        width = np.median(dists[dists > 0]) + 1e-10
    else:
        dists = np.sqrt(np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2))
        width = np.median(dists[dists > 0]) + 1e-10
    
    def make_features(X_):
        if d == 1:
            x_ = X_.flatten()
            c_ = centers.flatten()
            return np.exp(-0.5 * (x_[:, None] - c_[None, :]) ** 2 / width ** 2)
        else:
            dists_ = np.sqrt(np.sum((X_[:, None, :] - centers[None, :, :]) ** 2, axis=2))
            return np.exp(-0.5 * dists_ ** 2 / width ** 2)
    
    F_train = make_features(X)
    F_train = np.column_stack([np.ones(n), F_train])
    w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
    
    def predict(X_):
        F_test = make_features(X_)
        F_test = np.column_stack([np.ones(len(X_)), F_test])
        return F_test @ w
    
    return predict


def fit(X_train, y_train):
    """Main fit function: ensemble of multiple models selected by hold-out validation."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    n = len(y_train)
    d = X_train.shape[1]
    
    # Hold-out split for validation
    rng = np.random.RandomState(123)
    perm = rng.permutation(n)
    val_size = max(20, n // 5)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    
    # Candidate models: (name, predict_func, val_nmse)
    candidates = []
    
    # 1. Polynomials of various degrees
    for degree in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        if time.time() > deadline - 30:
            break
        try:
            model = _fit_polynomial(X_tr, y_tr, degree)
            preds = _safe_predict(X_val, model)
            nmse_val = _nmse(preds, y_val)
            candidates.append((f'poly_{degree}', model, nmse_val))
        except Exception:
            pass
    
    # 2. Splines with different smoothing
    for s_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        if time.time() > deadline - 30:
            break
        try:
            model = _fit_spline(X_tr, y_tr, s_factor)
            preds = _safe_predict(X_val, model)
            nmse_val = _nmse(preds, y_val)
            candidates.append((f'spline_{s_factor}', model, nmse_val))
        except Exception:
            pass
    
    # 3. RBF networks with different numbers of centers
    for nc in [10, 20, 30, 50]:
        if time.time() > deadline - 30:
            break
        try:
            model = _fit_rbf_network(X_tr, y_tr, nc)
            preds = _safe_predict(X_val, model)
            nmse_val = _nmse(preds, y_val)
            candidates.append((f'rbf_{nc}', model, nmse_val))
        except Exception:
            pass
    
    # 4. Gaussian process
    if time.time() < deadline - 60:
        try:
            model = _fit_gaussian_process(X_tr, y_tr)
            preds = _safe_predict(X_val, model)
            nmse_val = _nmse(preds, y_val)
            candidates.append(('gp', model, nmse_val))
        except Exception:
            pass
    
    # If no candidates, use degree-2 polynomial on all data
    if not candidates:
        model = _fit_polynomial(X_train, y_train, 2)
        return lambda X: _safe_predict(X, model)
    
    # Select best by validation NMSE
    candidates.sort(key=lambda x: x[2])
    best_name, best_model, best_nmse = candidates[0]
    
    # If best model is good enough, use it
    # Otherwise, consider ensemble of top-k
    if best_nmse < 0.01:
        # Very good fit, use single best
        # Refit on all data if possible
        # For now, just use the trained model
        return lambda X: _safe_predict(X, best_model)
    
    # Ensemble top-3 models weighted by inverse NMSE
    top_k = min(3, len(candidates))
    top_models = candidates[:top_k]
    weights = np.array([1.0 / (c[2] + 1e-10) for c in top_models])
    weights = weights / weights.sum()
    
    def predict(X):
        preds = np.zeros(len(X))
        for w, (_, model, _) in zip(weights, top_models):
            preds += w * _safe_predict(X, model)
        return preds
    
    return predict
# EVOLVE-BLOCK-END
