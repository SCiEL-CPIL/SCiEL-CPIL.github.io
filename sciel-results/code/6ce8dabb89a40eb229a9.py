"""
Strategy: Fixed-Knot Cubic Spline Model.
The model is:
f(x) = sum_{i=0}^{9} c_i * B_i(x)

Where B_i(x) are cubic B-spline basis functions defined on the interval [-2, 2]
with 13 uniformly spaced knots.

Parameters (length 10):
params[0:10] : Coefficients c_0 to c_9 for the spline basis.

This approach is linear in the parameters, making the optimization problem convex.
BFGS will reliably find the global minimum for the given basis, avoiding the
local minima issues associated with the non-linear width parameter in the
current RBF-Polynomial strategy. Cubic splines provide C2 continuity and
local flexibility, making them a strong universal approximator for smooth
functions on bounded domains.
"""
import numpy as np
from scipy.interpolate import BSpline

# Pre-compute the knot vector for the cubic spline.
# We want 10 coefficients.
# For BSpline, n_coeffs = n_knots - degree.
# degree = 3.
# n_knots = 10 + 3 = 13.
# Uniform knots on [-2, 2].
KNOTS = np.linspace(-2.0, 2.0, 13)
DEGREE = 3

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    try:
        # Create the spline object
        # BSpline evaluates the spline defined by knots, coefficients, and degree
        spline = BSpline(KNOTS, params, DEGREE)
        
        # Evaluate at x
        result = spline(x)
        
        # Ensure finite output
        result = np.where(np.isfinite(result), result, 0.0)
        
        return result
    except Exception:
        # Fallback to zero if anything goes wrong
        return np.zeros_like(x)
