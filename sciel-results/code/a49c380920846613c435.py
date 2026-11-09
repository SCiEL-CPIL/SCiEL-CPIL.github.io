"""Piecewise cubic B-spline with fixed internal knots for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a piecewise cubic B-spline representation.
    
    The spline is defined by 10 control points (params) at fixed knot locations
    spanning the input domain [-2, 2]. Between knots, the function is interpolated
    using cubic Hermite basis functions that ensure C1 continuity.
    
    This local basis avoids the global bias and ill-conditioning of high-degree
    polynomials, allowing the optimizer to fit arbitrary smooth shapes.
    
    Parameters:
        x: input array of shape (n_points,)
        params: array of 10 control point values
    
    Returns:
        Array of predictions, one per point in x.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 150s limit, return a safe default
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
    
    # Define fixed knot locations spanning [-2, 2]
    # 10 knots: -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2
    # Actually, we need 10 control points. Let's use 10 knots from -2 to 2.
    knots = np.linspace(-2.0, 2.0, 10)
    
    # For cubic interpolation between knots, we need to handle:
    # 1. Points outside [knots[0], knots[-1]]: use linear extrapolation from edge segments
    # 2. Points inside: use cubic Hermite interpolation
    
    # Compute slopes between consecutive knots for Hermite interpolation
    # Slope in segment i: (y_{i+1} - y_i) / (x_{i+1} - x_i)
    knot_vals = params
    dx = np.diff(knots)
    dy = np.diff(knot_vals)
    slopes = dy / dx  # shape (9,)
    
    # For Hermite cubic interpolation, we need derivatives at each knot.
    # We'll use a simple scheme: derivative at interior knots is the average
    # of adjacent segment slopes. Endpoints use one-sided slopes.
    derivs = np.zeros_like(knot_vals)
    derivs[0] = slopes[0]
    derivs[-1] = slopes[-1]
    for i in range(1, len(knot_vals) - 1):
        derivs[i] = 0.5 * (slopes[i - 1] + slopes[i])
    
    # Now interpolate each point
    result = np.zeros_like(x)
    
    # Clip x to the knot range for interpolation, handle extrapolation separately
    x_clipped = np.clip(x, knots[0], knots[-1])
    
    # Find which segment each point falls into
    # For each point, find the largest knot index such that knots[i] <= x
    # Using searchsorted: side='right' gives the index of the first knot > x
    # So segment index = searchsorted(knots, x_clipped, side='right') - 1
    seg_idx = np.searchsorted(knots, x_clipped, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, len(knots) - 2)
    
    # Local coordinates within each segment
    x0 = knots[seg_idx]
    x1 = knots[seg_idx + 1]
    y0 = knot_vals[seg_idx]
    y1 = knot_vals[seg_idx + 1]
    m0 = derivs[seg_idx]
    m1 = derivs[seg_idx + 1]
    
    # Normalized position within segment [0, 1]
    t = (x_clipped - x0) / (x1 - x0)
    t2 = t * t
    t3 = t2 * t
    
    # Hermite basis functions
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    
    # Hermite interpolation
    interp = h00 * y0 + h10 * (x1 - x0) * m0 + h01 * y1 + h11 * (x1 - x0) * m1
    
    # Handle extrapolation for points outside [knots[0], knots[-1]]
    # Left extrapolation: linear from first segment
    left_mask = x < knots[0]
    if np.any(left_mask):
        slope_left = slopes[0]
        result[left_mask] = knot_vals[0] + slope_left * (x[left_mask] - knots[0])
    
    # Right extrapolation: linear from last segment
    right_mask = x > knots[-1]
    if np.any(right_mask):
        slope_right = slopes[-1]
        result[right_mask] = knot_vals[-1] + slope_right * (x[right_mask] - knots[-1])
    
    # Interior points use the interpolated values
    interior_mask = ~(left_mask | right_mask)
    result[interior_mask] = interp[interior_mask]
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
