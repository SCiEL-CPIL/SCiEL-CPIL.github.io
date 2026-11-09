"""Combined symbolic regression: Rich Basis + CV Ridge vs Spline Interpolation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import UnivariateSpline

def _build_features(X, max_p=10):
    """Build a rich feature matrix from input X (n, d)."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n, dtype=np.float64)]
    
    for j in range(d):
        x = X[:, j].astype(np.float64)
        ax = np.abs(x)
        
        # Polynomials x^1 .. x^max_p
        powv = x.copy()
        for p in range(1, max_p + 1):
            cols.append(powv)
            if p < max_p:
                powv = powv * x
        
        # Abs and sqrt of abs
        cols.append(ax)
        sqrt_ax = np.sqrt(ax)
        cols.append(sqrt_ax)
        cols.append(sqrt_ax * x)  # x * |x|^0.5
        cols.append(ax ** 0.25)
        
        # Reciprocal-ish (guarded)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(np.abs(x) > 1e-6, 1.0 / x, 0.0)
        cols.append(inv)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv2 = np.where(np.abs(x) > 1e-6, 1.0 / (x * x), 0.0)
        cols.append(inv2)
        
        # Trigonometric
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.sin(x * x))
        cols.append(np.cos(x * x))
        cols.append(np.sin(np.sqrt(ax)))
        
        # Exponential (clipped to avoid overflow)
        xc = np.clip(x, -50, 50)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        cols.append(np.exp(-x * x))  # Gaussian-like
        cols.append(np.exp(x * x * 0.5))
        
        # Logarithms (guarded)
        cols.append(np.where(x > 1e-9, np.log(x), 0.0))
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.where(ax > 1e-9, np.log(ax), 0.0))
        
        # Power laws (non-integer)
        cols.append(x * np.sqrt(ax))
        cols.append(x * ax * sqrt_ax)  # x |x|^1.5
        cols.append(np.sign(x) * (ax ** 0.75))
        cols.append(np.sign(x) * (ax ** 1.25))
        cols.append(np.sign(x) * (ax ** 0.5))
        cols.append(np.sign(x) * (ax ** 1.5))
        cols.append(np.sign(x) * (ax ** 2.5))
        
        # sinh, cosh, sech
        sc = np.clip(xc, -20, 20)
        cols.append(np.sinh(sc))
        cols.append(np.cosh(sc))
        with np.errstate(over='ignore'):
            sech = 1.0 / np.cosh(sc)
            sech[~np.isfinite(sech)] = 0.0
        cols.append(sech)
    
    # Pairwise interactions for multi-dim (d is small here)
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)

def _clean_features(Phi):
    """Remove non-finite and low-variance columns."""
    finite_cols = np.all(np.isfinite(Phi), axis=0)
    Phi_clean = Phi[:, finite_cols]
    if Phi_clean.shape[1] == 0:
        return Phi, np.zeros(Phi.shape[1], dtype=bool)
    col_var = np.var(Phi_clean, axis=0)
    keep = col_var > 1e-12
    return Phi_clean[:, keep], keep

def _ridge_solve(Phi, y, lam):
    """Solve ridge regression for given lambda."""
    nfeat = Phi.shape[1]
    if lam == 0.0:
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    else:
        A = Phi.T @ Phi + lam * np.eye(nfeat)
        b = Phi.T @ y
        try:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L.T, np.linalg.solve(L, b))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _nrmse(y, yhat):
    """Compute normalized MSE."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-12:
        return mse
    return mse / vy

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 290.0  # safety margin under 300s
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    best_model_type = None
    best_model_data = None
    best_cv_score = np.inf
    
    # --- Model 1: Rich Basis + Ridge ---
    try:
        Phi = _build_features(X_train, max_p=10)
        Phi_clean, keep_mask = _clean_features(Phi)
        
        if Phi_clean.shape[1] > 0:
            # Cross-validation for lambda selection
            best_w = None
            best_ridge_cv_score = np.inf
            best_lambda = None
            
            lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            
            # Simple 5-fold CV
            n_folds = 5
            fold_size = n // n_folds
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            
            for lam in lambdas:
                if time.time() > DEADLINE - 10:
                    break
                cv_scores = []
                try:
                    for fold in range(n_folds):
                        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
                        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                        
                        Phi_tr = Phi_clean[train_idx]
                        y_tr = y_train[train_idx]
                        Phi_val = Phi_clean[val_idx]
                        y_val = y_train[val_idx]
                        
                        w = _ridge_solve(Phi_tr, y_tr, lam)
                        yhat_val = Phi_val @ w
                        if not np.all(np.isfinite(yhat_val)):
                            cv_scores.append(np.inf)
                            continue
                        cv_scores.append(_nrmse(y_val, yhat_val))
                    
                    mean_cv = np.mean(cv_scores)
                    if mean_cv < best_ridge_cv_score:
                        best_ridge_cv_score = mean_cv
                        best_lambda = lam
                except Exception:
                    continue
            
            if best_lambda is not None:
                # Final fit with best lambda
                best_w = _ridge_solve(Phi_clean, y_train, best_lambda)
                best_w = np.where(np.isfinite(best_w), best_w, 0.0)
                
                if best_ridge_cv_score < best_cv_score:
                    best_cv_score = best_ridge_cv_score
                    best_model_type = 'ridge'
                    best_model_data = {
                        'w': best_w,
                        'keep_mask': keep_mask,
                        'y_mean': float(np.mean(y_train))
                    }
    except Exception:
        pass
    
    # --- Model 2: Spline (only for 1D inputs) ---
    if d == 1:
        try:
            # Sort by x for spline fitting
            x = X_train[:, 0]
            y = y_train
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # Handle duplicates by averaging or jittering if necessary
            # For CV, we need to be careful.
            # Let's do 5-fold CV for Spline as well.
            n_folds = 5
            fold_size = n // n_folds
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            
            best_spline_cv_score = np.inf
            best_spline_k = 3
            best_spline_s = 0.0
            
            # Try a few splines: k=1,2,3 with s=0 (interpolation) and maybe small s
            # Since data is noiseless, s=0 is usually best, but high k can oscillate.
            for k in [1, 2, 3]:
                if time.time() > DEADLINE - 10:
                    break
                cv_scores = []
                for fold in range(n_folds):
                    val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
                    train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                    
                    x_tr = x_sorted[train_idx] # Note: x_sorted is already sorted, but train_idx refers to original indices?
                    # Wait, x_sorted is sorted by x. But indices are random permutations of original rows.
                    # We need to sort the training subset by x for UnivariateSpline.
                    
                    x_tr_orig = X_train[train_idx, 0]
                    y_tr = y_train[train_idx]
                    sort_tr = np.argsort(x_tr_orig)
                    x_tr_s = x_tr_orig[sort_tr]
                    y_tr_s = y_tr[sort_tr]
                    
                    x_val = X_train[val_idx, 0]
                    y_val = y_train[val_idx]
                    
                    try:
                        # Check for duplicates in training set
                        if len(np.unique(x_tr_s)) < len(x_tr_s):
                            # Add tiny jitter
                            x_tr_s = x_tr_s + np.random.uniform(0, 1e-9, len(x_tr_s))
                        
                        spline = UnivariateSpline(x_tr_s, y_tr_s, s=0, k=k)
                        yhat_val = spline(x_val)
                        if not np.all(np.isfinite(yhat_val)):
                            cv_scores.append(np.inf)
                        else:
                            cv_scores.append(_nrmse(y_val, yhat_val))
                    except Exception:
                        cv_scores.append(np.inf)
                
                mean_cv = np.mean(cv_scores)
                if mean_cv < best_spline_cv_score:
                    best_spline_cv_score = mean_cv
                    best_spline_k = k
            
            if best_spline_cv_score < best_cv_score:
                # Refit with full data using best k
                try:
                    if len(np.unique(x_sorted)) < len(x_sorted):
                        x_sorted = x_sorted + np.random.uniform(0, 1e-9, len(x_sorted))
                    spline = UnivariateSpline(x_sorted, y_sorted, s=0, k=best_spline_k)
                    best_cv_score = best_spline_cv_score
                    best_model_type = 'spline'
                    best_model_data = {
                        'spline': spline
                    }
                except Exception:
                    pass
        except Exception:
            pass
    
    # --- Fallback: Mean ---
    if best_model_type is None:
        m = float(np.mean(y_train))
        return lambda X: np.full(X.shape[0], m)
    
    y_mean = float(np.mean(y_train))
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        n_test = X.shape[0]
        
        if best_model_type == 'ridge':
            try:
                P = _build_features(X, max_p=10)
                P_clean = P[:, best_model_data['keep_mask']]
                yhat = P_clean @ best_model_data['w']
                yhat = np.where(np.isfinite(yhat), yhat, best_model_data['y_mean'])
                return yhat.astype(np.float64)
            except Exception:
                return np.full(n_test, best_model_data['y_mean'], dtype=np.float64)
        
        elif best_model_type == 'spline':
            try:
                x = X[:, 0]
                yhat = best_model_data['spline'](x)
                yhat = np.where(np.isfinite(yhat), yhat, y_mean)
                return yhat.astype(np.float64)
            except Exception:
                return np.full(n_test, y_mean, dtype=np.float64)
        
        else:
            return np.full(n_test, y_mean, dtype=np.float64)
    
    return predict

# EVOLVE-BLOCK-END
