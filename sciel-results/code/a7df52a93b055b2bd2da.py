import numpy as np
import time
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import warnings
warnings.filterwarnings("ignore")

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 330.0
    margin = 5.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    # Ensure 1D input handling. If d>1, project to 1D via PCA or first col
    d = X.shape[1]
    if d > 1:
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        if S[0] / (S.sum() + 1e-12) > 0.99:
            X = (X_centered @ Vt[0]).reshape(-1, 1)
        else:
            X = X[:, :1]
            
    x = X.flatten()
    n = len(x)
    
    # Handle edge case: constant x
    if np.std(x) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            return np.full(X_test.shape[0], mean_y)
        return predict_const

    # Normalize x to [-1, 1] for Chebyshev stability
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2.0 * (x - x_min) / x_range - 1.0

    # Sort for interpolation methods
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_sorted = y[sort_idx]
    
    # Holdout split for model selection
    # Use a deterministic split based on index to avoid randomness issues in eval
    n_val = max(10, int(0.2 * n))
    # Interleaved split to ensure train/val have similar distribution
    val_mask = (np.arange(n) % 5 == 0)
    val_mask[:n_val] = True # Ensure at least n_val points if possible, though interleaving is better
    # Let's stick to a simple random-like shuffle for better generalization estimate
    indices = np.arange(n)
    # Simple deterministic shuffle
    np.random.seed(42)
    np.random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    x_train_val = x_norm[train_idx]
    y_train_val = y[train_idx]
    x_val_val = x_norm[val_idx]
    y_val_val = y[val_idx]

    best_model_type = None
    best_val_mse = np.inf
    best_predict_fn = None

    # Helper to evaluate model on validation set
    def evaluate_model(predict_fn, x_eval, y_eval):
        try:
            pred = predict_fn(x_eval.reshape(-1, 1))
            pred = np.asarray(pred, dtype=np.float64).flatten()
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y_eval) ** 2)
            return mse
        except Exception:
            return np.inf

    # --- Model Family 1: Chebyshev Spectral Approximation ---
    # We fit a Chebyshev series of degree N.
    # y(x) = sum_{k=0}^{N} a_k T_k(t)
    # We solve for a using Ridge regression to handle ill-conditioning for high N.
    
    cheb_degree_max = 64 # Max degree to try
    # Precompute Chebyshev values for training and validation sets
    # Using numpy.polynomial.chebyshev.chebv
    from numpy.polynomial import chebyshev as C
    
    # Generate design matrix for training data
    # C.chebv(x, deg) returns (deg+1, len(x)) -> we want (len(x), deg+1)
    # Actually, chebv(x, deg) where x is (N,) returns (deg+1, N).
    
    candidate_cheb_models = []
    
    for deg in range(2, cheb_degree_max + 1):
        if time.time() > deadline - margin:
            break
            
        # Design matrix T_train[i, j] = T_j(x_train[i])
        # C.chebv returns coefficients evaluated at points.
        # T_vals = C.chebv(x_train_val, deg) -> shape (deg+1, N_train)
        T_train = C.chebv(x_train_val, deg).T
        T_val = C.chebv(x_val_val, deg).T
        
        # Solve Ridge Regression: (T^T T + lambda I) a = T^T y
        # We use a small lambda to stabilize high degrees.
        # Since data is noiseless, we want near-exact fit but regularized to avoid oscillation.
        # Let's try a few lambda values or just use lstsq with rcond.
        # Better: Use SVD to truncate small singular values effectively.
        
        try:
            # Standardize y? No, Chebyshev is orthogonal, so scaling is good.
            # Use np.linalg.lstsq for stability, or explicit ridge.
            # Let's use a small ridge alpha that increases slightly with degree?
            # Or just solve least squares.
            # For noiseless data, if the function is a polynomial of degree P, 
            # Chebyshev fit of degree P should be exact (up to numerical precision).
            
            # Use Ridge to prevent overfitting to numerical noise if any.
            alpha = 1e-8 
            A = T_train.T @ T_train + alpha * np.eye(deg + 1)
            b = T_train.T @ y_train_val
            a_coeffs = np.linalg.solve(A, b)
            
            # Predict
            pred_train = T_train @ a_coeffs
            pred_val = T_val @ a_coeffs
            
            mse_val = np.mean((pred_val - y_val_val) ** 2)
            
            if mse_val < best_val_mse:
                # Store the coefficients and degree
                # We need to refit on ALL data later, so just store params
                best_val_mse = mse_val
                best_model_type = 'chebyshev'
                best_cheb_deg = deg
                # We don't store the predict fn yet, we refit on full data
            else:
                # Early stopping heuristic: if MSE is not decreasing, stop?
                # Chebyshev coefficients should decay. If they don't, function might be non-smooth.
                pass
                
        except Exception:
            continue

    # --- Model Family 2: Smooth Spline (UnivariateSpline) ---
    # Splines are good for non-smooth or complex shapes where polynomial approximation is slow.
    
    best_spline_s = None
    best_spline_mse = np.inf
    
    # Search over smoothing parameter s
    # s = sum((y - spline(x))^2). Lower s = closer fit.
    # For noiseless data, we want s close to 0 but not 0 to avoid oscillation.
    # However, UnivariateSpline with s=0 interpolates exactly.
    # Let's try a range of s values relative to variance.
    y_var = np.var(y_train_val)
    s_candidates = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    # Also try s proportional to N
    s_candidates.extend([n * 1e-6, n * 1e-4, n * 1e-2])
    
    for s_val in s_candidates:
        if time.time() > deadline - margin:
            break
        try:
            # Use sorted data for spline
            spl = UnivariateSpline(x_sorted[train_idx], y_sorted[train_idx], s=s_val, k=4)
            pred_val = spl(x_val_val)
            mse_val = np.mean((pred_val - y_val_val) ** 2)
            
            if mse_val < best_spline_mse:
                best_spline_mse = mse_val
                best_spline_s = s_val
        except Exception:
            continue
    
    if best_spline_s is not None and best_spline_mse < best_val_mse:
        best_val_mse = best_spline_mse
        best_model_type = 'spline'

    # --- Model Family 3: PCHIP (Preserve Monotonicity, Cubic Hermite) ---
    # Good if the function is monotonic or has sharp corners.
    
    try:
        # PCHIP requires strictly increasing x.
        # We have sorted x.
        pchip = PchipInterpolator(x_sorted[train_idx], y_sorted[train_idx])
        pred_val = pchip(x_val_val)
        mse_val = np.mean((pred_val - y_val_val) ** 2)
        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_model_type = 'pchip'
    except Exception:
        pass

    # --- Refit Best Model on All Data ---
    
    if best_model_type == 'chebyshev':
        deg = best_cheb_deg
        T_full = C.chebv(x_norm, deg).T
        alpha = 1e-8
        A = T_full.T @ T_full + alpha * np.eye(deg + 1)
        b = T_full.T @ y
        a_coeffs = np.linalg.solve(A, b)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()
            # Map to [-1, 1]
            if x_range < 1e-12:
                t_test = np.zeros_like(x_test)
            else:
                t_test = 2.0 * (x_test - x_min) / x_range - 1.0
            T_test = C.chebv(t_test, deg).T
            preds = T_test @ a_coeffs
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)

    elif best_model_type == 'spline':
        s_val = best_spline_s
        spl = UnivariateSpline(x_sorted, y_sorted, s=s_val, k=4)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()
            if x_range < 1e-12:
                return np.full(x_test.shape, spl(x_sorted[0]))
            t_test = 2.0 * (x_test - x_min) / x_range - 1.0
            preds = spl(t_test)
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)

    elif best_model_type == 'pchip':
        pchip = PchipInterpolator(x_sorted, y_sorted)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()
            if x_range < 1e-12:
                return np.full(x_test.shape, pchip(x_sorted[0]))
            t_test = 2.0 * (x_test - x_min) / x_range - 1.0
            preds = pchip(t_test)
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)

    else:
        # Fallback: Linear regression on x
        A = np.column_stack([np.ones(n), x_norm])
        b = y
        try:
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test.flatten()
                if x_range < 1e-12:
                    t_test = np.zeros_like(x_test)
                else:
                    t_test = 2.0 * (x_test - x_min) / x_range - 1.0
                A_test = np.column_stack([np.ones(len(t_test)), t_test])
                preds = A_test @ coeffs
                return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        except Exception:
            mean_y = np.mean(y)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                return np.full(X_test.shape[0], mean_y)

    return predict
