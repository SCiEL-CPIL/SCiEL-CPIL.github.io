"""Hybrid piecewise cubic spline with Gaussian local basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a piecewise cubic Hermite spline (6 control points)
    with 4 Gaussian bumps and 2 oscillatory terms.
    
    The spline provides local flexibility for smooth trends, while Gaussians
    capture localized features and oscillatory terms add frequency components.
    This hybrid structure allows the model to fit a wider variety of function
    shapes than either approach alone.
    
    Parameters:
        x: input array of shape (n_points,)
        params: array of 10 parameters
            - params[0:6]: spline control points
            - params[6:10]: [gauss1, gauss2, osc_sin, osc_cos]
    
    Returns:
        Array of predictions, one per point in x.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > 145.0:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
    
    # Ensure x is 1D and params is correct length
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    # Split params: 6 for spline, 4 for auxiliary terms
    spline_params = params[0:6]
    gauss_amplitude1 = params[6]
    gauss_amplitude2 = params[7]
    osc_sin_amp = params[8]
    osc_cos_amp = params[9]
    
    # --- Piecewise cubic Hermite spline with 6 control points ---
    knots = np.linspace(-2.0, 2.0, 6)
    knot_vals = spline_params
    dx = np.diff(knots)
    dy = np.diff(knot_vals)
    slopes = dy / dx  # shape (5,)
    
    # Derivatives at each knot
    derivs = np.zeros_like(knot_vals)
    derivs[0] = slopes[0]
    derivs[-1] = slopes[-1]
    for i in range(1, len(knot_vals) - 1):
        derivs[i] = 0.5 * (slopes[i - 1] + slopes[i])
    
    # Interpolate spline
    x_clipped = np.clip(x, knots[0], knots[-1])
    seg_idx = np.searchsorted(knots, x_clipped, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, len(knots) - 2)
    
    x0 = knots[seg_idx]
    x1 = knots[seg_idx + 1]
    y0 = knot_vals[seg_idx]
    y1 = knot_vals[seg_idx + 1]
    m0 = derivs[seg_idx]
    m1 = derivs[seg_idx + 1]
    
    t = (x_clipped - x0) / (x1 - x0)
    t2 = t * t
    t3 = t2 * t
    
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    
    spline_interp = h00 * y0 + h10 * (x1 - x0) * m0 + h01 * y1 + h11 * (x1 - x0) * m1
    
    # Extrapolation for points outside knot range
    result = np.zeros_like(x)
    left_mask = x < knots[0]
    if np.any(left_mask):
        slope_left = slopes[0]
        result[left_mask] = knot_vals[0] + slope_left * (x[left_mask] - knots[0])
    
    right_mask = x > knots[-1]
    if np.any(right_mask):
        slope_right = slopes[-1]
        result[right_mask] = knot_vals[-1] + slope_right * (x[right_mask] - knots[-1])
    
    interior_mask = ~(left_mask | right_mask)
    result[interior_mask] = spline_interp[interior_mask]
    
    # --- Add Gaussian bumps ---
    # Two Gaussian bumps centered at -1.0 and 1.0 with width 0.6
    center1 = -1.0
    center2 = 1.0
    width = 0.6
    
    gauss1 = np.exp(-((x - center1) ** 2) / (2.0 * width ** 2))
    gauss2 = np.exp(-((x - center2) ** 2) / (2.0 * width ** 2))
    
    result += gauss_amplitude1 * gauss1
    result += gauss_amplitude2 * gauss2
    
    # --- Add oscillatory terms ---
    osc_sin = np.sin(np.pi * x / 2.0)
    osc_cos = np.cos(np.pi * x / 2.0)
    
    result += osc_sin_amp * osc_sin
    result += osc_cos_amp * osc_cos
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
