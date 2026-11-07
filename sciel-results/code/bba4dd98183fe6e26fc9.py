# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.linalg import solve

# Global deadline
_TOTAL_DEADLINE = time.time() + 318.0

def _remaining():
    return _TOTAL_DEADLINE - time.time()

def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            # Clip to reasonable range to prevent overflow in scoring
            y = np.clip(y, -1e10, 1e10)
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict

def _fit_pchip(X, y):
    """Fit a PCHIP interpolator. Handles duplicates by averaging."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Handle duplicates: average y values for same x
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(unique_xs) < 4:
            return None

        s = PchipInterpolator(unique_xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None

def _fit_cubic_spline(X, y):
    """Fit a CubicSpline interpolator. Handles duplicates by averaging."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Handle duplicates: average y values for same x
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(unique_xs) < 4:
            return None

        s = CubicSpline(unique_xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None

def _poly_features(X, degree, center=True):
    """Polynomial features up to given degree. Centered for stability."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        if center:
            # Use training mean if provided via closure or global, but here we pass X
            # For safety, we center by the mean of the current X passed in.
            # Note: This is slightly different for train/test if means differ, 
            # but for 1D uniform ranges, it's a good heuristic.
            # Better: pass center value.
            pass 
        # We will use a simpler approach: just use x directly but clip powers?
        # Actually, let's use a fixed center based on the range [-3,3] approx 0.
        # Or better, pass the center from the training data mean.
        # Since _poly_features is called with X, and we want consistency, 
        # we should probably compute features relative to 0 or the global mean.
        # Let's stick to standard polynomial but ensure no overflow.
        for p in range(1, degree + 1):
            cols.append(x ** p)
    return np.column_stack(cols)

def _centered_poly_features(X, degree, center_val=0.0):
    """Centered polynomial features."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j] - center_val
        for p in range(1, degree + 1):
            cols.append(x ** p)
    return np.column_stack(cols)

def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 10):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
    return np.column_stack(cols)

def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        # Clip x to prevent exp overflow
        xc = np.clip(x, -10, 10)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        # Safe reciprocal
        cols.append(1.0 / (np.abs(x) + 0.1)) 
        cols.append(np.exp(-x**2))
    return np.column_stack(cols)

def _mixed_features(X):
    """Combine polynomial, trig, and exp features."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for p in range(1, 8):
            cols.append(x ** p)
        for k in range(1, 8):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        xc = np.clip(x, -10, 10)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 0.1))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
    return np.column_stack(cols)

def _fit_robust_linear(F, y, ridge_alpha=1e-6):
    """Solve least squares with ridge regularization using SVD for stability."""
    try:
        n, p = F.shape
        if not np.all(np.isfinite(F)) or not np.all(np.isfinite(y)):
            return None
        # Add ridge term to diagonal
        # We want to solve (F^T F + alpha I) w = F^T y
        # Using SVD of F is more stable
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        # Regularize singular values
        s_reg = s / (s**2 + ridge_alpha)
        # w = V^T U^T y * s_reg
        Ut_y = U.T @ y
        w = Vt.T @ (Ut_y * s_reg)
        
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None

def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    """Try fitting a linear model on given features."""
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        
        w = _fit_robust_linear(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        
        # Create a predict function that uses the same feature map
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Fp = feat_fn(Xp)
            Fp = np.nan_to_num(Fp, nan=0.0, posinf=1e6, neginf=-1e6)
            return Fp @ w
        return _safe_predict(predict), rmse
    except Exception:
        return None, np.inf

def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-4):
    """Kernel Ridge Regression with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        # Use solve, fallback to lstsq if singular
        try:
            w = solve(K, y)
        except Exception:
            w, *_ = np.linalg.lstsq(K, y, rcond=None)
            
        if not np.all(np.isfinite(w)):
            return None

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            Ks = np.exp(-0.5 * sq_dist)
            preds = Ks @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return _safe_predict(predict)
    except Exception:
        return None

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    # 1. Split data for validation
    # Use a deterministic split to ensure reproducibility
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_size = max(20, n // 5)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_trn, y_trn = X_train[train_idx], y_train[train_idx]
    
    # Fallback if split is bad
    if len(X_trn) < 10:
        X_trn, y_trn = X_train, y_train
        X_val, y_val = X_train, y_train

    best_predict = None
    best_val_rmse = np.inf
    best_model_type = None
    best_params = None

    # Baseline: Constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_val - const) ** 2))
    best_val_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))
    best_model_type = 'const'
    best_params = {'const': const}

    # 2. Evaluate Candidates on Validation Set
    
    # A. Splines (Very strong for noiseless 1D)
    if d == 1 and _remaining() > 5.0:
        # PCHIP
        pchip_pred = _fit_pchip(X_trn, y_trn)
        if pchip_pred is not None:
            pred = pchip_pred(X_val)
            rmse = np.sqrt(np.mean((pred - y_val) ** 2))
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_model_type = 'pchip'
                best_params = {} # Will refit later

        # Cubic Spline
        if _remaining() > 5.0:
            cs_pred = _fit_cubic_spline(X_trn, y_trn)
            if cs_pred is not None:
                pred = cs_pred(X_val)
                rmse = np.sqrt(np.mean((pred - y_val) ** 2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'cubic_spline'
                    best_params = {}

    # B. Linear Models with various feature sets
    if _remaining() > 10.0:
        # Polynomials
        for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
            if _remaining() < 2.0: break
            # Use centered polynomials for stability
            center_val = np.mean(X_trn[:, 0]) if d==1 else 0.0
            feat_fn = lambda Xp, d=deg, c=center_val: _centered_poly_features(Xp, d, c)
            predict, rmse = _try_fit_family(feat_fn, X_trn, y_trn)
            if predict is not None:
                pred = predict(X_val)
                val_rmse = np.sqrt(np.mean((pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model_type = 'poly'
                    best_params = {'degree': deg, 'center': center_val}

        # Trig
        if _remaining() > 5.0:
            predict, rmse = _try_fit_family(_trig_features, X_trn, y_trn)
            if predict is not None:
                pred = predict(X_val)
                val_rmse = np.sqrt(np.mean((pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model_type = 'trig'
                    best_params = {}

        # Exp/Log
        if _remaining() > 5.0:
            predict, rmse = _try_fit_family(_exp_features, X_trn, y_trn)
            if predict is not None:
                pred = predict(X_val)
                val_rmse = np.sqrt(np.mean((pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model_type = 'exp'
                    best_params = {}

        # Mixed
        if _remaining() > 5.0:
            predict, rmse = _try_fit_family(_mixed_features, X_trn, y_trn)
            if predict is not None:
                pred = predict(X_val)
                val_rmse = np.sqrt(np.mean((pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model_type = 'mixed'
                    best_params = {}

    # C. Kernel Methods
    if _remaining() > 15.0:
        for ls in [0.5, 1.0, 2.0, 5.0]:
            if _remaining() < 2.0: break
            kr_predict = _fit_kernel_ridge(X_trn, y_trn, length_scale=ls, alpha=1e-4)
            if kr_predict is not None:
                pred = kr_predict(X_val)
                rmse = np.sqrt(np.mean((pred - y_val) ** 2))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_model_type = 'kr'
                    best_params = {'length_scale': ls}

    # 3. Refit Best Model on Full Training Data
    if best_model_type != 'const' and _remaining() > 5.0:
        try:
            if best_model_type == 'pchip':
                final_pred = _fit_pchip(X_train, y_train)
                if final_pred is not None:
                    best_predict = final_pred
            elif best_model_type == 'cubic_spline':
                final_pred = _fit_cubic_spline(X_train, y_train)
                if final_pred is not None:
                    best_predict = final_pred
            elif best_model_type == 'poly':
                center_val = best_params['center']
                degree = best_params['degree']
                feat_fn = lambda Xp: _centered_poly_features(Xp, degree, center_val)
                final_pred, _ = _try_fit_family(feat_fn, X_train, y_train)
                if final_pred is not None:
                    best_predict = final_pred
            elif best_model_type == 'trig':
                final_pred, _ = _try_fit_family(_trig_features, X_train, y_train)
                if final_pred is not None:
                    best_predict = final_pred
            elif best_model_type == 'exp':
                final_pred, _ = _try_fit_family(_exp_features, X_train, y_train)
                if final_pred is not None:
                    best_predict = final_pred
            elif best_model_type == 'mixed':
                final_pred, _ = _try_fit_family(_mixed_features, X_train, y_train)
                if final_pred is not None:
                    best_predict = final_pred
            elif best_model_type == 'kr':
                ls = best_params['length_scale']
                final_pred = _fit_kernel_ridge(X_train, y_train, length_scale=ls, alpha=1e-4)
                if final_pred is not None:
                    best_predict = final_pred
        except Exception:
            pass # Keep previous best_predict

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
