"""Multi-kernel ensemble: full RBF KRR + polynomial + linear, CV-selected."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _rbf_kernel(A, B, length_scale):
    """Compute RBF kernel between sets A and B."""
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * length_scale**2))

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    if i + j + k == deg:
                        cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _kfold_cv_predict(X, y, predict_fn, n_folds=5, seed=0):
    """K-fold cross-validation normalized MSE."""
    n = len(y)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    cv_scores = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        try:
            pred_val = predict_fn(X_tr, y_tr, X_val)
            mse = np.mean((pred_val - y_val)**2)
            var_y = np.var(y)
            if var_y > 0:
                cv_scores.append(mse / var_y)
            else:
                cv_scores.append(mse)
        except Exception:
            cv_scores.append(np.inf)
    return np.mean(cv_scores)

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 310.0  # Safety margin from 330s limit
    
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
    
    # Median distance for RBF length scale
    # Use subsample for speed if n > 50
    if n > 100:
        rng_sub = np.random.RandomState(0)
        idx_sub = rng_sub.choice(n, size=100, replace=False)
        dists_sub = np.sqrt(np.sum((X_norm[idx_sub][:, None, :] - X_norm[idx_sub][None, :, :])**2, axis=2))
        np.fill_diagonal(dists_sub, np.inf)
        median_dist = np.median(dists_sub[dists_sub > 0])
    else:
        dists_full = np.sqrt(np.sum((X_norm[:, None, :] - X_norm[None, :, :])**2, axis=2))
        np.fill_diagonal(dists_full, np.inf)
        median_dist = np.median(dists_full[dists_full > 0])
    
    if median_dist == 0 or np.isnan(median_dist):
        median_dist = 1.0
    
    # Candidate models: (type, params)
    # RBF KRR with various length scales and regularizations
    length_scales = [median_dist * 0.2, median_dist * 0.5, median_dist, median_dist * 2.0, median_dist * 4.0]
    rbf_lambdas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    
    # Polynomial degrees
    poly_degrees = [2, 3, 4]
    poly_lambdas = [1e-6, 1e-4, 1e-2]
    
    # Linear
    linear_lambdas = [1e-8, 1e-6, 1e-4]
    
    # Collect all candidate models with their CV scores
    candidates = []  # list of (cv_score, model_type, params, predict_fn_factory)
    
    # --- RBF KRR candidates ---
    for ls in length_scales:
        if time.time() > DEADLINE:
            break
        K_train = _rbf_kernel(X_norm, X_norm, ls)
        for lam in rbf_lambdas:
            if time.time() > DEADLINE:
                break
            try:
                A = K_train + lam * np.eye(n)
                alpha = np.linalg.solve(A, y)
            except np.linalg.LinAlgError:
                continue
            
            # CV score using leave-few-out
            def rbf_predict(X_tr, y_tr, X_val, ls=ls, lam=lam, x_min=x_min, x_range=x_range):
                Xn = (X_val - x_min) / x_range
                K_val = _rbf_kernel(Xn, Xn, ls)
                # Refit on training fold
                K_tr = _rbf_kernel((X_tr - x_min) / x_range, (X_tr - x_min) / x_range, ls)
                try:
                    a = np.linalg.solve(K_tr + lam * np.eye(len(y_tr)), y_tr)
                except np.linalg.LinAlgError:
                    a = np.linalg.pinv(K_tr + lam * np.eye(len(y_tr))) @ y_tr
                # K_val is X_val @ X_val, but we need K(X_val, X_train)
                K_cross = _rbf_kernel(Xn, (X_tr - x_min) / x_range, ls)
                return K_cross @ a
            
            cv_score = _kfold_cv_predict(X_norm, y, rbf_predict, n_folds=5, seed=42)
            candidates.append((cv_score, 'rbf', (ls, lam), rbf_predict))
    
    # --- Polynomial candidates ---
    for deg in poly_degrees:
        if time.time() > DEADLINE:
            break
        # Precompute feature matrix
        try:
            F_train = _poly_features(X_norm, deg)
        except Exception:
            continue
        
        for lam in poly_lambdas:
            if time.time() > DEADLINE:
                break
            try:
                # Ridge regression
                A = F_train.T @ F_train + lam * np.eye(F_train.shape[1])
                b = F_train.T @ y
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            def poly_predict(X_tr, y_tr, X_val, deg=deg, lam=lam, x_min=x_min, x_range=x_range):
                Xn_val = (X_val - x_min) / x_range
                Xn_tr = (X_tr - x_min) / x_range
                F_val = _poly_features(Xn_val, deg)
                F_tr = _poly_features(Xn_tr, deg)
                try:
                    A = F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])
                    b = F_tr.T @ y_tr
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w = np.linalg.pinv(F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])) @ F_tr.T @ y_tr
                return F_val @ w
            
            cv_score = _kfold_cv_predict(X_norm, y, poly_predict, n_folds=5, seed=42)
            candidates.append((cv_score, 'poly', (deg, lam), poly_predict))
    
    # --- Linear candidate ---
    for lam in linear_lambdas:
        if time.time() > DEADLINE:
            break
        def lin_predict(X_tr, y_tr, X_val, lam=lam, x_min=x_min, x_range=x_range):
            Xn_val = (X_val - x_min) / x_range
            Xn_tr = (X_tr - x_min) / x_range
            # Add bias
            F_val = np.column_stack([np.ones(len(Xn_val)), Xn_val])
            F_tr = np.column_stack([np.ones(len(Xn_tr)), Xn_tr])
            try:
                A = F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])
                b = F_tr.T @ y_tr
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1])) @ F_tr.T @ y_tr
            return F_val @ w
        
        cv_score = _kfold_cv_predict(X_norm, y, lin_predict, n_folds=5, seed=42)
        candidates.append((cv_score, 'linear', (lam,), lin_predict))
    
    # Sort by CV score
    candidates.sort(key=lambda x: x[0])
    
    # Take top-K models (K=3) for ensemble
    K_ensemble = min(3, len(candidates))
    top_models = candidates[:K_ensemble]
    
    # Build ensemble predict function
    # Each model needs to be refit on full training data
    ensemble_models = []
    for cv_score, mtype, params, predict_fn in top_models:
        if mtype == 'rbf':
            ls, lam = params
            K_train = _rbf_kernel(X_norm, X_norm, ls)
            try:
                alpha = np.linalg.solve(K_train + lam * np.eye(n), y)
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(K_train + lam * np.eye(n)) @ y
            ensemble_models.append(('rbf', ls, alpha))
        elif mtype == 'poly':
            deg, lam = params
            F_train = _poly_features(X_norm, deg)
            try:
                A = F_train.T @ F_train + lam * np.eye(F_train.shape[1])
                b = F_train.T @ y
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(F_train.T @ F_train + lam * np.eye(F_train.shape[1])) @ F_train.T @ y
            ensemble_models.append(('poly', deg, w))
        elif mtype == 'linear':
            lam = params[0]
            F_train = np.column_stack([np.ones(n), X_norm])
            try:
                A = F_train.T @ F_train + lam * np.eye(F_train.shape[1])
                b = F_train.T @ y
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(F_train.T @ F_train + lam * np.eye(F_train.shape[1])) @ F_train.T @ y
            ensemble_models.append(('linear', 0, w))
    
    # If no valid models, fallback to linear
    if not ensemble_models:
        F_train = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            Xn = (X - x_min) / x_range
            Fn = np.column_stack([np.ones(len(Xn)), Xn])
            return Fn @ w
        return predict
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        Xn = (X - x_min) / x_range
        preds = []
        for mtype, param, params_data in ensemble_models:
            if mtype == 'rbf':
                ls, alpha = param, params_data
                K_val = _rbf_kernel(Xn, X_norm, ls)
                preds.append(K_val @ alpha)
            elif mtype == 'poly':
                deg, w = param, params_data
                F_val = _poly_features(Xn, deg)
                preds.append(F_val @ w)
            elif mtype == 'linear':
                _, w = param, params_data
                F_val = np.column_stack([np.ones(len(Xn)), Xn])
                preds.append(F_val @ w)
        # Average predictions
        pred_avg = np.mean(preds, axis=0)
        return pred_avg
    
    return predict

# EVOLVE-BLOCK-END
