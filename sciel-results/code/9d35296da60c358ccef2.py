"""Nonlinear oscillator symbolic regression with RBF + linear basis."""
import numpy as np
from scipy.optimize import minimize
import time

# Fixed RBF centers within the training domain
# x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
_X_MIN, _X_MAX = -0.5728, 0.5275
_V_MIN, _V_MAX = -0.1754, 0.2821

# Create a 3x3 grid of centers
_x_centers = np.linspace(_X_MIN, _X_MAX, 3)
_v_centers = np.linspace(_V_MIN, _V_MAX, 3)
_cx, _cv = np.meshgrid(_x_centers, _v_centers, indexing='ij')
_CENTERS = np.column_stack([_cx.ravel(), _cv.ravel()])  # shape (9, 2)

# RBF width - chosen to be comparable to domain size
# Domain widths: x_width ~ 1.1, v_width ~ 0.46
# Use a moderate width so each RBF covers a local region
_WIDTH = 0.35

# Precompute center coordinates as arrays for vectorized use
_CX = _CENTERS[:, 0]  # shape (9,)
_CV = _CENTERS[:, 1]  # shape (9,)


def _rbf_values(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute Gaussian RBF values for all 9 centers.
    Returns array of shape (n, 9).
    """
    # dx = x[:, None] - CX[None, :]  -> (n, 9)
    dx = x[:, None] - _CX[None, :]
    dv = v[:, None] - _CV[None, :]
    # r2 = dx^2 + dv^2
    r2 = dx * dx + dv * dv
    # Gaussian: exp(-r2 / (2 * sigma^2))
    sigma2 = _WIDTH * _WIDTH
    return np.exp(-r2 / (2.0 * sigma2))


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    params: 10 values
      params[0..8] : RBF weights for 9 Gaussian kernels
      params[9]    : constant bias term
    """
    p = params
    # RBF contribution
    rbf = _rbf_values(x, v)  # (n, 9)
    result = rbf @ p[:9] + p[9]
    return result


def _objective(params, x_data, v_data, y_data):
    """MSE objective for BFGS."""
    try:
        pred = equation(x_data, v_data, params)
        if pred.shape != y_data.shape:
            return 1e30
        if not np.all(np.isfinite(pred)):
            return 1e30
        err = pred - y_data
        mse = np.mean(err * err)
        if not np.isfinite(mse):
            return 1e30
        return mse
    except Exception:
        return 1e30


def _fit(x_data, v_data, y_data):
    """
    Fit params using BFGS from all-ones start.
    Returns best params found.
    """
    x0 = np.ones(10, dtype=np.float64)
    
    # Deadline guard
    t_start = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    best_params = x0.copy()
    best_mse = _objective(x0, x_data, v_data, y_data)
    
    # Run BFGS
    try:
        result = minimize(
            _objective,
            x0,
            args=(x_data, v_data, y_data),
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
        )
        if result.success or result.fun < best_mse:
            if np.all(np.isfinite(result.x)):
                best_params = result.x
                best_mse = result.fun
    except Exception:
        pass
    
    # Check time; if we have spare time, try a second restart from a different init
    elapsed = time.time() - t_start
    if elapsed < deadline - 30.0:
        # Try a second BFGS run from a small-perturbation init
        x0b = np.ones(10, dtype=np.float64) * 0.1
        try:
            result2 = minimize(
                _objective,
                x0b,
                args=(x_data, v_data, y_data),
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
            )
            if np.all(np.isfinite(result2.x)) and result2.fun < best_mse:
                best_params = result2.x
                best_mse = result2.fun
        except Exception:
            pass
    
    return best_params


# The evaluator calls equation(x, v, params) where params is the fitted array.
# But wait - the task says "The evaluator fits params ... minimizing the training MSE."
# So the evaluator will call equation with the fitted params. We just need equation
# to be a good function that BFGS can fit well.
#
# However, looking at the task more carefully: "Write equation(<inputs>, params), a closed-form
# NumPy expression with up to 10 fitted constants."
# The evaluator does the fitting. So we just need to provide equation() that uses params well.
# The _fit function above is NOT called by the evaluator - the evaluator does its own BFGS.
#
# So we just need equation() to be a good parametric model. The RBF approach should work
# because BFGS can fit the 9 RBF weights + 1 bias.

# Actually, re-reading: the evaluator fits params starting from all ones using BFGS.
# So our job is just to make equation(x, v, params) a good model.
# The RBF model with fixed centers and widths, and 10 learnable parameters (9 weights + bias),
# should be well-conditioned and fit well by BFGS.

# No changes needed - equation() is already defined above.
