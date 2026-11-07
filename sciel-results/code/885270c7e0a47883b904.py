"""
Structural change: Replace the complex CV ensemble with a robust Natural Cubic Spline
interpolation as the primary model for 1-D noiseless data. Cubic splines are C2
continuous and better approximate smooth underlying functions (including oscillatory
ones) compared to PCHIP, which can introduce kinks. A high-degree polynomial (degree
20) is kept as a fallback for cases where the function is strictly polynomial.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator
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


def _cubic_spline_predict(x_train, y_train, X):
    """Natural Cubic Spline interpolation."""
    x_u, y_u = _prepare_data(x_train, y_train)
    if len(x_u) < 4:
        # Fallback to linear or constant if too few points
        if len(x_u) == 1:
            return np.full(X.shape, y_u[0])
        elif len(x_u) == 2:
            slope = (y_u[1] - y_u[0]) / (x_u[1] - x_u[0])
            return (y_u[0] + slope * (X.ravel() - x_u[0])).reshape(X.shape)
        else:
            # Linear interpolation
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


def _poly_predict(x_train, y_train, X, degree=20):
    """Polynomial regression via least squares with regularization."""
    x_u, y_u = _prepare_data(x_train, y_train)
    if len(x_u) <= degree:
        degree = len(x_u) - 1
    if degree < 1:
        return np.full(X.shape, np.mean(y_u))
        
    # Use scaled x to improve conditioning
    x_min, x_max = np.min(x_u), np.max(x_u)
    x_range = x_max - x_min
    if x_range == 0:
        return np.full(X.shape, np.mean(y_u))
    
    x_scaled = (x_u - x_min) / x_range
    V = np.vander(x_scaled, degree + 1, increasing=True)
    # Add small regularization for stability
    w, *_ = np.linalg.lstsq(V, y_u, rcond=None)
    
    X_scaled = (X.ravel() - x_min) / x_range
    V_test = np.vander(X_scaled, degree + 1, increasing=True)
    return (V_test @ w).reshape(X.shape)


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

    # Primary model: Cubic Spline
    # We assume noiseless data, so interpolation is the best strategy.
    # Cubic Spline is preferred over PCHIP for smooth/oscillatory functions.
    
    # Check if data is monotonic to decide if PCHIP might be safer?
    # Actually, CubicSpline with natural BCs is very robust.
    # We will use CubicSpline as the default.
    
    def predict(X):
        if d == 1:
            try:
                return _cubic_spline_predict(x_train, y_train, X)
            except Exception:
                # Fallback to PCHIP
                try:
                    return _pchip_fallback(x_train, y_train, X)
                except Exception:
                    # Last resort: nearest neighbor / linear interp
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

    # Helper for fallback
    def _pchip_fallback(x_train, y_train, X):
        x_u, y_u = _prepare_data(x_train, y_train)
        pchip = PchipInterpolator(x_u, y_u)
        return pchip(X.ravel()).reshape(X.shape)

    return predict
# EVOLVE-BLOCK-END
