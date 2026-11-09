"""Combined symbolic regression: Robust 1D interpolation + robust basis regression with outlier handling."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

def _nrmse(y, yhat):
    """Compute normalized MSE."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-12:
        return mse
    return mse / vy

def _build_features(X, max_p=12):
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

def _make_spline_predictor(x_train, y_train, s=0, k=3):
    """Create a spline predictor with outlier detection and fallback."""
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    has_duplicates = len(unique_x) < len(x_sorted)
    
    if has_duplicates:
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
    else:
        unique_x = x_sorted
        y_unique = y_sorted
    
    try:
        spline = UnivariateSpline(unique_x, y_unique, s=s, k=k)
        
        # Validate on training data
        yhat_train = spline(x_train)
        if not np.all(np.isfinite(yhat_train)):
            return None
        
        train_score = _nrmse(y_train, yhat_train)
        
        # Check for outliers: if train score is high, try with fewer points
        if train_score > 1e-4:
            # Try removing top 5% outliers based on residual
            resid = np.abs(y_train - yhat_train)
            threshold = np.percentile(resid, 95)
            keep_mask = resid <= threshold
            if np.sum(keep_mask) >= 10:
                try:
                    spline2 = UnivariateSpline(x_train[keep_mask], y_train[keep_mask], s=s, k=k)
                    yhat_train2 = spline2(x_train)
                    if np.all(np.isfinite(yhat_train2)):
                        train_score2 = _nrmse(y_train, yhat_train2)
                        if train_score2 < train_score:
                            spline = spline2
                            train_score = train_score2
                except Exception:
                    pass
        
        y_mean = float(np.mean(y_train))
        
        def spline_predict(X, spline=spline, y_mean=y_mean):
            x_test = X[:, 0]
            y_pred = spline(x_test)
            return np.where(np.isfinite(y_pred), y_pred, y_mean).astype(np.float64)
        
        return spline_predict, train_score
    except Exception:
        return None

def _make_linear_interp_predictor(x_train, y_train):
    """Create a linear interpolation predictor with outlier detection."""
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    has_duplicates = len(unique_x) < len(x_sorted)
    
    if has_duplicates:
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
    else:
        unique_x = x_sorted
        y_unique = y_sorted
    
    try:
        interp_func = interp1d(unique_x, y_unique, kind='linear', fill_value='extrapolate')
        yhat_train = interp_func(x_train)
        
        if not np.all(np.isfinite(yhat_train)):
            return None
        
        train_score = _nrmse(y_train, yhat_train)
        
        y_mean = float(np.mean(y_train))
        
        def interp_predict(X, interp_func=interp_func, y_mean=y_mean):
            x_test = X[:, 0]
            y_pred = interp_func(x_test)
            return np.where(np.isfinite(y_pred), y_pred, y_mean).astype(np.float64)
        
        return interp_predict, train_score
    except Exception:
        return None

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 285.0  # safety margin under 300s
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    y_mean = float(np.mean(y_train))
    
    best_predict = None
    best_score = np.inf
    best_method = None
    
    # For 1D: try interpolation methods first (most likely to work for smooth functions)
    if d == 1:
        x_train = X_train[:, 0]
        
        # Method 1: Cubic spline interpolation (s=0 for noiseless)
        result = _make_spline_predictor(x_train, y_train, s=0, k=3)
        if result is not None:
            predict_fn, score = result
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_method = 'spline_s0'
                best_predict = predict_fn
        
        # Method 2: Spline with small smoothing (handles slight noise)
        if best_score > 1e-6:
            for s_val in [1e-8, 1e-6, 1e-4]:
                if time.time() > DEADLINE - 30:
                    break
                result = _make_spline_predictor(x_train, y_train, s=s_val, k=3)
                if result is not None:
                    predict_fn, score = result
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_method = f'spline_s{s_val}'
                        best_predict = predict_fn
        
        # Method 3: Linear interpolation (robust fallback)
        if best_score > 1e-4:
            result = _make_linear_interp_predictor(x_train, y_train)
            if result is not None:
                predict_fn, score = result
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_method = 'linear_interp'
                    best_predict = predict_fn
        
        # Method 4: Quadratic spline
        if best_score > 1e-4:
            result = _make_spline_predictor(x_train, y_train, s=0, k=2)
            if result is not None:
                predict_fn, score = result
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_method = 'spline_k2'
                    best_predict = predict_fn
    
    # Fallback: Rich basis + CV Ridge
    if best_predict is None or best_score > 1e-3:
        try:
            Phi = _build_features(X_train, max_p=12)
            Phi_clean, keep_mask = _clean_features(Phi)
            
            if Phi_clean.shape[1] > 0:
                best_w = None
                best_cv_score = np.inf
                best_lambda = None
                
                lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
                
                n_folds = 5
                fold_size = n // n_folds
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                
                for lam in lambdas:
                    if time.time() > DEADLINE - 15:
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
                        if mean_cv < best_cv_score:
                            best_cv_score = mean_cv
                            best_lambda = lam
                    except Exception:
                        continue
                
                if best_lambda is not None:
                    best_w = _ridge_solve(Phi_clean, y_train, best_lambda)
                    best_w = np.where(np.isfinite(best_w), best_w, 0.0)
                    
                    yhat_train = Phi_clean @ best_w
                    yhat_train = np.where(np.isfinite(yhat_train), yhat_train, 0.0)
                    train_score = _nrmse(y_train, yhat_train)
                    
                    if np.isfinite(train_score) and (best_predict is None or train_score < best_score):
                        w_final = best_w
                        mask_final = keep_mask
                        
                        def ridge_predict(X, w=w_final, mask=mask_final, y_mean=y_mean):
                            P = _build_features(X, max_p=12)
                            P_clean = P[:, mask]
                            yhat = P_clean @ w
                            return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                        
                        best_predict = ridge_predict
                        best_score = train_score
                        best_method = 'ridge'
        except Exception:
            pass
    
    # Final fallback: mean
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    return best_predict

# EVOLVE-BLOCK-END
