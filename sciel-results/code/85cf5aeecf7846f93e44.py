"""
Symbolic regression via multi-kernel ensemble with leave-one-out selection.

Combines:
- Full RBF Kernel Ridge Regression (fast, flexible)
- Polynomial features + Ridge (captures algebraic structure)
- Matern-5/2 GP-style kernel (sharper local adaptation)
- Rational function approximator (numerator/denominator polynomials)

Selects the single best model per equation using leave-one-out CV,
with a safety deadline guard.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _rbf_kernel(A, B, length_scale):
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * length_scale**2))

def _matern52_kernel(A, B, length_scale):
    """Matern-5/2 kernel: (1 + sqrt(3)*d + 3d^2/2) * exp(-sqrt(3)*d)"""
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    d = np.sqrt(sq_dists)
    sqrt3 = np.sqrt(3.0)
    return (1.0 + sqrt3 * d + 1.5 * d**2) * np.exp(-sqrt3 * d / length_scale)

def _poly_features(X, degree):
    """All polynomial monomials up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        # generate all combinations with repetition of length deg
        from itertools import combinations_with_replacement
        for combo in combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for idx in combo:
                feat = feat * X[:, idx]
            cols.append(feat)
    return np.column_stack(cols)

def _rational_features(X, degree_num, degree_den):
    """
    Rational function: f(x) = P(x) / Q(x) where P is degree_num poly, Q is degree_den poly.
    We linearize by multiplying: y * Q(x) = P(x)
    So we solve: P(x) - y * Q(x) = 0
    Features: [P_terms, -y * Q_terms] but Q_terms depend on y...
    Actually, for a linear-in-parameters rational model:
    f(x) = sum_i a_i phi_i(x) / (1 + sum_j b_j psi_j(x))
    This is nonlinear in params. Instead, use:
    f(x) = sum_i a_i phi_i(x) / sum_j c_j psi_j(x)
    where psi includes constant. This is still nonlinear.
    
    Simpler approach: fit P(x) and Q(x) separately via iterative reweighting,
    or just use a high-degree polynomial which approximates rational functions.
    
    For simplicity, skip rational and use high-degree polynomial instead.
    """
    return None

def _fit_rbf_krr(X_norm, y, ls, lam):
    """Fit full RBF KRR. Returns (alpha, predict_fn)."""
    n = len(y)
    K = _rbf_kernel(X_norm, X_norm, ls)
    A = K + lam * np.eye(n)
    try:
        alpha = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(A) @ y
    
    def predict(X_test, alpha=alpha, ls=ls, x_min=x_min, x_range=x_range):
        Xn = (X_test - x_min) / x_range
        K_test = _rbf_kernel(Xn, X_norm, ls)
        return K_test @ alpha
    
    return alpha, predict

def _fit_poly_ridge(X_norm, y, degree, lam):
    """Fit polynomial ridge regression."""
    F = _poly_features(X_norm, degree)
    A = F.T @ F + lam * np.eye(F.shape[1])
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    
    def predict(X_test, w=w, degree=degree, x_min=x_min, x_range=x_range):
        Xn = (X_test - x_min) / x_range
        F_test = _poly_features(Xn, degree)
        return F_test @ w
    
    return w, predict

def _fit_matern_krr(X_norm, y, ls, lam):
    """Fit full Matern-5/2 KRR."""
    n = len(y)
    K = _matern52_kernel(X_norm, X_norm, ls)
    A = K + lam * np.eye(n)
    try:
        alpha = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(A) @ y
    
    def predict(X_test, alpha=alpha, ls=ls, x_min=x_min, x_range=x_range):
        Xn = (X_test - x_min) / x_range
        K_test = _matern52_kernel(Xn, X_norm, ls)
        return K_test @ alpha
    
    return alpha, predict

def _loo_cv_score(model_predict_fn, X_norm, y, n_folds=10, seed=0):
    """K-fold CV normalized MSE for a model that takes (X_train, y_train, X_test)."""
    n = len(y)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    mse_sum = 0.0
    count = 0
    var_y = np.var(y)
    if var_y == 0:
        var_y = 1.0
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_tr, y_tr = X_norm[train_idx], y[train_idx]
        X_val = X_norm[val_idx]
        y_val = y[val_idx]
        
        try:
            pred_val = model_predict_fn(X_tr, y_tr, X_val)
            mse = np.mean((pred_val - y_val)**2)
            mse_sum += mse
            count += 1
        except Exception:
            mse_sum += np.inf
            count += 1
    
    if count == 0:
        return np.inf
    return (mse_sum / count) / var_y

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 315.0  # Safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    X_norm = (X - x_min) / x_range
    
    var_y = np.var(y)
    if var_y == 0:
        var_y = 1.0
    
    # Compute median distance for length scale heuristics
    if n > 100:
        rng_sub = np.random.RandomState(0)
        idx_sub = rng_sub.choice(n, size=100, replace=False)
        X_sub = X_norm[idx_sub]
        dists_sub = np.sqrt(np.sum((X_sub[:, None, :] - X_sub[None, :, :])**2, axis=2))
        np.fill_diagonal(dists_sub, np.inf)
        median_dist = np.median(dists_sub[dists_sub > 0])
    else:
        dists_full = np.sqrt(np.sum((X_norm[:, None, :] - X_norm[None, :, :])**2, axis=2))
        np.fill_diagonal(dists_full, np.inf)
        median_dist = np.median(dists_full[dists_full > 0])
    
    if median_dist == 0 or np.isnan(median_dist):
        median_dist = 1.0
    
    # Candidate hyperparameters
    rbf_ls = [median_dist * 0.2, median_dist * 0.5, median_dist, median_dist * 2.0, median_dist * 4.0]
    rbf_lams = [1e-8, 1e-6, 1e-4, 1e-2]
    
    matern_ls = [median_dist * 0.3, median_dist * 0.7, median_dist, median_dist * 2.0]
    matern_lams = [1e-8, 1e-6, 1e-4, 1e-2]
    
    poly_degrees = [2, 3, 4, 5]
    poly_lams = [1e-6, 1e-4, 1e-2]
    
    candidates = []  # (cv_score, model_type, params, predict_fn)
    
    # --- RBF KRR candidates ---
    for ls in rbf_ls:
        if time.time() > DEADLINE:
            break
        for lam in rbf_lams:
            if time.time() > DEADLINE:
                break
            try:
                K = _rbf_kernel(X_norm, X_norm, ls)
                A = K + lam * np.eye(n)
                alpha = np.linalg.solve(A, y)
                
                # LOO CV using hat matrix trick for KRR:
                # LOO prediction: y_i - alpha_i / (1 + lam) ... not exactly.
                # Use K-fold CV instead for correctness.
                def rbf_cv_predict(X_tr, y_tr, X_val, ls=ls, lam=lam, x_min=x_min, x_range=x_range):
                    Xn_tr = (X_tr - x_min) / x_range
                    Xn_val = (X_val - x_min) / x_range
                    K_tr = _rbf_kernel(Xn_tr, Xn_tr, ls)
                    A_tr = K_tr + lam * np.eye(len(y_tr))
                    try:
                        a = np.linalg.solve(A_tr, y_tr)
                    except np.linalg.LinAlgError:
                        a = np.linalg.pinv(A_tr) @ y_tr
                    K_cross = _rbf_kernel(Xn_val, Xn_tr, ls)
                    return K_cross @ a
                
                cv_score = _loo_cv_score(rbf_cv_predict, X_norm, y, n_folds=10, seed=42)
                
                # Final predict using full data
                def rbf_predict(X_test, alpha=alpha, ls=ls, x_min=x_min, x_range=x_range):
                    Xn = (X_test - x_min) / x_range
                    K_test = _rbf_kernel(Xn, X_norm, ls)
                    return K_test @ alpha
                
                candidates.append((cv_score, 'rbf', (ls, lam), rbf_predict))
            except Exception:
                continue
    
    # --- Matern KRR candidates ---
    for ls in matern_ls:
        if time.time() > DEADLINE:
            break
        for lam in matern_lams:
            if time.time() > DEADLINE:
                break
            try:
                K = _matern52_kernel(X_norm, X_norm, ls)
                A = K + lam * np.eye(n)
                alpha = np.linalg.solve(A, y)
                
                def matern_cv_predict(X_tr, y_tr, X_val, ls=ls, lam=lam, x_min=x_min, x_range=x_range):
                    Xn_tr = (X_tr - x_min) / x_range
                    Xn_val = (X_val - x_min) / x_range
                    K_tr = _matern52_kernel(Xn_tr, Xn_tr, ls)
                    A_tr = K_tr + lam * np.eye(len(y_tr))
                    try:
                        a = np.linalg.solve(A_tr, y_tr)
                    except np.linalg.LinAlgError:
                        a = np.linalg.pinv(A_tr) @ y_tr
                    K_cross = _matern52_kernel(Xn_val, Xn_tr, ls)
                    return K_cross @ a
                
                cv_score = _loo_cv_score(matern_cv_predict, X_norm, y, n_folds=10, seed=42)
                
                def matern_predict(X_test, alpha=alpha, ls=ls, x_min=x_min, x_range=x_range):
                    Xn = (X_test - x_min) / x_range
                    K_test = _matern52_kernel(Xn, X_norm, ls)
                    return K_test @ alpha
                
                candidates.append((cv_score, 'matern', (ls, lam), matern_predict))
            except Exception:
                continue
    
    # --- Polynomial Ridge candidates ---
    for deg in poly_degrees:
        if time.time() > DEADLINE:
            break
        try:
            F_train = _poly_features(X_norm, deg)
        except Exception:
            continue
        
        for lam in poly_lams:
            if time.time() > DEADLINE:
                break
            try:
                A = F_train.T @ F_train + lam * np.eye(F_train.shape[1])
                b = F_train.T @ y
                w = np.linalg.solve(A, b)
                
                def poly_cv_predict(X_tr, y_tr, X_val, deg=deg, lam=lam, x_min=x_min, x_range=x_range):
                    Xn_tr = (X_tr - x_min) / x_range
                    Xn_val = (X_val - x_min) / x_range
                    F_tr = _poly_features(Xn_tr, deg)
                    F_val = _poly_features(Xn_val, deg)
                    A_tr = F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])
                    b_tr = F_tr.T @ y_tr
                    try:
                        w_tr = np.linalg.solve(A_tr, b_tr)
                    except np.linalg.LinAlgError:
                        w_tr = np.linalg.pinv(A_tr) @ b_tr
                    return F_val @ w_tr
                
                cv_score = _loo_cv_score(poly_cv_predict, X_norm, y, n_folds=10, seed=42)
                
                def poly_predict(X_test, w=w, deg=deg, x_min=x_min, x_range=x_range):
                    Xn = (X_test - x_min) / x_range
                    F_test = _poly_features(Xn, deg)
                    return F_test @ w
                
                candidates.append((cv_score, 'poly', (deg, lam), poly_predict))
            except Exception:
                continue
    
    # --- Linear fallback ---
    try:
        F_lin = np.column_stack([np.ones(n), X_norm])
        w_lin = np.linalg.lstsq(F_lin, y, rcond=None)[0]
        def lin_cv_predict(X_tr, y_tr, X_val, x_min=x_min, x_range=x_range):
            Xn_tr = (X_tr - x_min) / x_range
            Xn_val = (X_val - x_min) / x_range
            F_tr = np.column_stack([np.ones(len(Xn_tr)), Xn_tr])
            F_val = np.column_stack([np.ones(len(Xn_val)), Xn_val])
            w_tr = np.linalg.lstsq(F_tr, y_tr, rcond=None)[0]
            return F_val @ w_tr
        
        cv_score = _loo_cv_score(lin_cv_predict, X_norm, y, n_folds=10, seed=42)
        
        def lin_predict(X_test, w=w_lin, x_min=x_min, x_range=x_range):
            Xn = (X_test - x_min) / x_range
            F_test = np.column_stack([np.ones(len(Xn)), Xn])
            return F_test @ w
        
        candidates.append((cv_score, 'linear', (), lin_predict))
    except Exception:
        pass
    
    # Sort by CV score
    candidates.sort(key=lambda x: x[0])
    
    if len(candidates) == 0:
        # Ultimate fallback: constant
        y_mean = np.mean(y)
        def predict_const(X):
            return np.full(len(X), y_mean)
        return predict_const
    
    # Return the best model
    best_score, best_type, best_params, best_predict = candidates[0]
    
    # Wrap to ensure finite output
    def safe_predict(X):
        X = np.asarray(X, dtype=np.float64)
        try:
            pred = best_predict(X)
            pred = np.asarray(pred, dtype=np.float64)
            pred = np.where(np.isfinite(pred), pred, np.mean(y))
            return pred
        except Exception:
            return np.full(len(X), np.mean(y))
    
    return safe_predict

# EVOLVE-BLOCK-END
