"""
Structural change: Use a Fourier series regression as the primary model for 1-D
noiseless data, leveraging the ability to capture oscillatory components that
PCHIP and polynomials may miss or oscillate on. Use regularized least-squares
with adaptive frequency cutoffs to prevent overfitting. Combine with Cubic Spline
and PCHIP as fallbacks, selected via cross-validation.

The key insight is that Fourier series are particularly good at representing
smooth periodic-like functions and can handle oscillatory behavior better than
polynomial interpolation which suffers from Runge phenomenon.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# Global deadline: 320 s (10 s safety margin before 330 s kill)
_DEADLINE = time.time() + 320.0


def _elapsed_ok():
    return time.time() < _DEADLINE


def _prepare_data(x_train, y_train):
    """Sort and clean 1D data for interpolation."""
    x = x_train.ravel()
    y = y_train.ravel()
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Remove duplicate x values (keep first)
    mask = np.concatenate(([True], np.diff(x_sorted) > 1e-12))
    x_u = x_sorted[mask]
    y_u = y_sorted[mask]
    return x_u, y_u


def _fourier_predict(x_train, y_train, X, max_freq=20, reg=1e-8):
    """Fourier series regression with adaptive frequency cutoff."""
    x_u, y_u = _prepare_data(x_train, y_train)
    if len(x_u) < 4:
        if len(x_u) == 1:
            return np.full(X.shape, y_u[0])
        elif len(x_u) == 2:
            slope = (y_u[1] - y_u[0]) / (x_u[1] - x_u[0])
            return (y_u[0] + slope * (X.ravel() - x_u[0])).reshape(X.shape)
        else:
            return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
    
    x_min, x_max = np.min(x_u), np.max(x_u)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.full(X.shape, np.mean(y_u))
    
    # Scale to [0, 2*pi] for Fourier basis
    x_scaled = (x_u - x_min) / x_range * 2 * np.pi
    X_scaled = (X.ravel() - x_min) / x_range * 2 * np.pi
    
    # Build Fourier basis matrix
    n = len(x_scaled)
    freqs = np.arange(0, max_freq + 1)
    
    # Start with constant term
    V = np.ones((n, 1))
    
    for k in freqs[1:]:
        V = np.column_stack([V, np.sin(k * x_scaled), np.cos(k * x_scaled)])
    
    # Regularized least squares
    V_test = np.ones((len(X_scaled), 1))
    for k in freqs[1:]:
        V_test = np.column_stack([V_test, np.sin(k * X_scaled), np.cos(k * X_scaled)])
    
    try:
        # Add regularization
        A = V.T @ V + reg * np.eye(V.shape[1])
        b = V.T @ y_u
        w = np.linalg.solve(A, b)
        pred = V_test @ w
        return pred.reshape(X.shape)
    except np.linalg.LinAlgError:
        try:
            w, *_ = np.linalg.lstsq(V, y_u, rcond=None)
            pred = V_test @ w
            return pred.reshape(X.shape)
        except Exception:
            # Fallback to PCHIP
            try:
                pchip = PchipInterpolator(x_u, y_u)
                return pchip(X.ravel()).reshape(X.shape)
            except Exception:
                return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)


def _pchip_predict(x_train, y_train, X):
    """PCHIP interpolation on sorted 1-D data."""
    x_u, y_u = _prepare_data(x_train, y_train)
    if len(x_u) < 2:
        return np.full(X.shape, y_u[0] if len(x_u) == 1 else np.mean(y_u))
    try:
        pchip = PchipInterpolator(x_u, y_u)
        return pchip(X.ravel()).reshape(X.shape)
    except Exception:
        return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)


def _cubic_spline_predict(x_train, y_train, X):
    """Natural Cubic Spline interpolation."""
    x_u, y_u = _prepare_data(x_train, y_train)
    if len(x_u) < 4:
        if len(x_u) == 1:
            return np.full(X.shape, y_u[0])
        elif len(x_u) == 2:
            slope = (y_u[1] - y_u[0]) / (x_u[1] - x_u[0])
            return (y_u[0] + slope * (X.ravel() - x_u[0])).reshape(X.shape)
        else:
            return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
    
    try:
        spl = CubicSpline(x_u, y_u, bc_type='natural')
        return spl(X.ravel()).reshape(X.shape)
    except Exception:
        # Fallback to PCHIP if spline fails
        try:
            pchip = PchipInterpolator(x_u, y_u)
            return pchip(X.ravel()).reshape(X.shape)
        except Exception:
            return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)


def _poly_predict(x_train, y_train, X, degree=12):
    """Polynomial regression via least squares with scaling."""
    x_u, y_u = _prepare_data(x_train, y_train)
    if len(x_u) <= degree:
        degree = len(x_u) - 1
    if degree < 1:
        return np.full(X.shape, np.mean(y_u))
        
    x_min, x_max = np.min(x_u), np.max(x_u)
    x_range = x_max - x_min
    if x_range == 0:
        return np.full(X.shape, np.mean(y_u))
    
    x_scaled = (x_u - x_min) / x_range
    V = np.vander(x_scaled, degree + 1, increasing=True)
    w, *_ = np.linalg.lstsq(V, y_u, rcond=None)
    
    X_scaled = (X.ravel() - x_min) / x_range
    V_test = np.vander(X_scaled, degree + 1, increasing=True)
    return (V_test @ w).reshape(X.shape)


def _gp_predict(x_train, y_train, X):
    """Gaussian Process Regression with Matern 5/2 kernel for 1-D noiseless data."""
    x_u, y_u = _prepare_data(x_train, y_train)
    n = len(x_u)
    if n < 2:
        return np.full(X.shape, y_u[0] if n == 1 else np.mean(y_u))
    
    # Standardize x for better numerical stability
    x_mean = np.mean(x_u)
    x_std = np.std(x_u)
    if x_std < 1e-12:
        return np.full(X.shape, y_u[0])
    
    x_u_std = (x_u - x_mean) / x_std
    X_std = (X.ravel() - x_mean) / x_std
    
    # Standardize y
    y_mean = np.mean(y_u)
    y_std = np.std(y_u)
    if y_std < 1e-12:
        return np.full(X.shape, y_mean)
    
    y_u_std = (y_u - y_mean) / y_std
    
    def matern52_kernel(x1, x2, length_scale, variance):
        """Matern 5/2 kernel."""
        r = np.sqrt((x1[:, None] - x2[None, :]) ** 2)
        sqrt3_r = np.sqrt(3) * r / length_scale
        K = variance * (1 + sqrt3_r + sqrt3_r ** 2 / 3) * np.exp(-sqrt3_r)
        return K
    
    def log_likelihood(params):
        """Negative log marginal likelihood for GP hyperparameters."""
        try:
            length_scale = np.exp(params[0])
            variance = np.exp(params[1])
            noise = np.exp(params[2])
            
            K = matern52_kernel(x_u_std, x_u_std, length_scale, variance)
            K += noise * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_u_std))
            
            log_det = 2 * np.sum(np.log(np.diag(L)))
            ll = 0.5 * y_u_std @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
            return ll + 1e-6 * length_scale  # Small regularization
        
        except Exception:
            return 1e10
    
    # Optimize hyperparameters
    try:
        # Initial guess: length_scale=1, variance=1, noise=1e-6
        x0 = np.array([0.0, 0.0, -10.0])
        result = minimize(log_likelihood, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'ftol': 1e-6})
        
        length_scale = np.exp(result.x[0])
        variance = np.exp(result.x[1])
        noise = np.exp(result.x[2])
        
        # Compute predictions
        K = matern52_kernel(x_u_std, x_u_std, length_scale, variance)
        K += noise * np.eye(n)
        
        K_s = matern52_kernel(x_u_std, X_std, length_scale, variance)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_u_std))
            v = np.linalg.solve(L, K_s.T)
            
            mean_std = (K_s @ alpha).ravel()
            pred_std = mean_std * y_std + y_mean
            return pred_std.reshape(X.shape)
            
        except np.linalg.LinAlgError:
            # Fallback to PCHIP
            try:
                pchip = PchipInterpolator(x_u, y_u)
                return pchip(X.ravel()).reshape(X.shape)
            except Exception:
                return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
                
    except Exception:
        # Fallback to PCHIP
        try:
            pchip = PchipInterpolator(x_u, y_u)
            return pchip(X.ravel()).reshape(X.shape)
        except Exception:
            return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)


def _cv_score(model_fn, x_train, y_train, n_folds=5):
    """K-fold CV MSE on 1-D data."""
    x_u, y_u = _prepare_data(x_train, y_train)
    n = len(x_u)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    total_mse = 0.0
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        trn_idx = np.concatenate([indices[:start], indices[end:]])
        x_tr = x_u[trn_idx]
        y_tr = y_u[trn_idx]
        x_val = x_u[val_idx]
        y_val = y_u[val_idx]
        try:
            pred = model_fn(x_tr.reshape(-1, 1), y_tr.reshape(-1, 1), x_val.reshape(-1, 1))
            mse = np.mean((pred - y_val) ** 2)
            total_mse += mse
        except Exception:
            total_mse += np.inf
    return total_mse / n_folds


def fit(X_train, y_train):
    t_start = time.time()
    X = X_train
    y = y_train

    # All equations are 1-D per the task contract
    d = X.shape[1]
    if d == 1:
        x_train = X.ravel()
        y_train_1d = y.ravel()
    else:
        # Fallback for multi-dim: use polynomial
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        return lambda X: features(X) @ w

    # Candidate models with their hyperparameters
    candidates = []

    # Fourier series with different max frequencies
    for max_freq in [10, 20, 30]:
        candidates.append((f"fourier_{max_freq}", 
                          lambda xt, yt, Xq, mf=max_freq: _fourier_predict(xt, yt, Xq, max_freq=mf)))

    # PCHIP (primary fallback)
    candidates.append(("pchip", lambda xt, yt, Xq: _pchip_predict(xt, yt, Xq)))

    # Cubic Spline
    candidates.append(("spline", lambda xt, yt, Xq: _cubic_spline_predict(xt, yt, Xq)))

    # Polynomial degrees
    for deg in [8, 12, 16]:
        candidates.append((f"poly{deg}", lambda xt, yt, Xq, d=deg: _poly_predict(xt, yt, Xq, d)))
    
    # Gaussian Process with Matern 5/2 kernel
    candidates.append(("gp_matern52", lambda xt, yt, Xq: _gp_predict(xt, yt, Xq)))

    # Evaluate candidates via CV, but respect deadline
    best_model = None
    best_cv = np.inf
    best_name = "fourier_20"

    # Default: GP with Matern 5/2 kernel (better for smooth 1-D noiseless data)
    try:
        default_pred = _gp_predict(x_train, y_train, X_train)
    except Exception:
        default_pred = _fourier_predict(x_train, y_train, X_train, max_freq=20)

    for name, fn in candidates:
        if not _elapsed_ok():
            break
        try:
            cv = _cv_score(fn, x_train, y_train, n_folds=5)
            if cv < best_cv:
                best_cv = cv
                best_model = fn
                best_name = name
        except Exception:
            pass

    # If deadline is close, use GP or Fourier fallback
    if time.time() > _DEADLINE - 30:
        try:
            best_model = lambda xt, yt, Xq: _gp_predict(xt, yt, Xq)
            best_name = "gp_fallback"
        except Exception:
            best_model = lambda xt, yt, Xq: _fourier_predict(xt, yt, Xq, max_freq=20)
            best_name = "fourier_fallback"

    # Final prediction function
    def predict(X):
        if d == 1:
            try:
                result = best_model(x_train, y_train, X)
                # Ensure finite predictions
                if not np.all(np.isfinite(result)):
                    # Replace non-finite values with interpolation
                    x_u, y_u = _prepare_data(x_train, y_train)
                    interp = np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
                    result = np.where(np.isfinite(result), result, interp)
                return result
            except Exception:
                # Fallback to GP
                try:
                    result = _gp_predict(x_train, y_train, X)
                    if not np.all(np.isfinite(result)):
                        x_u, y_u = _prepare_data(x_train, y_train)
                        interp = np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
                        result = np.where(np.isfinite(result), result, interp)
                    return result
                except Exception:
                    # Fallback to PCHIP
                    try:
                        result = _pchip_predict(x_train, y_train, X)
                        if not np.all(np.isfinite(result)):
                            x_u, y_u = _prepare_data(x_train, y_train)
                            interp = np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
                            result = np.where(np.isfinite(result), result, interp)
                        return result
                    except Exception:
                        # Last resort: linear interpolation
                        x_u, y_u = _prepare_data(x_train, y_train)
                        return np.interp(X.ravel(), x_u, y_u).reshape(X.shape)
        else:
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            F = np.column_stack(cols)
            return F @ w

    return predict
# EVOLVE-BLOCK-END
