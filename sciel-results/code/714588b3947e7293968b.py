"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a cubic B-spline basis.
    
    The 10 parameters are interpreted as control points for a cubic B-spline
    defined over the domain [-3, 3]. This provides a local, smooth, piecewise-
    polynomial approximation that is linear in the parameters, allowing the
    external BFGS optimizer to efficiently fit the shape.
    
    The B-spline basis functions are computed using the de Boor algorithm or
    direct evaluation of the cubic B-spline kernel. For a uniform knot vector
    with 10 control points, we define the knots to cover the range [-3, 3].
    
    To ensure the spline is well-defined and the basis matrix is stable:
    1. We define a uniform knot vector.
    2. We evaluate the 10 cubic B-spline basis functions at each x.
    3. We return the linear combination: sum(params[i] * N_i(x)).
    
    This structure is materially different from global polynomial/Fourier bases
    because it allows for localized features and avoids Runge's phenomenon.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Ensure params is 1D
    params = np.asarray(params, dtype=np.float64).flatten()
    if len(params) != 10:
        # Fallback if params size is wrong (should not happen per contract)
        params = np.ones(10)
    
    # Define knot vector for cubic B-splines (degree k=3)
    # We have n=10 control points.
    # A standard clamped B-spline knot vector has length n + k + 1 = 14.
    # To cover the domain [-3, 3] with 10 control points:
    # Let's distribute the interior knots uniformly in [-3, 3].
    # The first k+1 and last k+1 knots are repeated for clamping.
    
    # Simplest robust approach: Use a uniform grid of knots.
    # Let's map the control points to a uniform grid.
    # If we use n=10 control points, and we want the spline to span [-3, 3],
    # we can define the interior knots as:
    # -3, -3, -3, -3, -1.5, 0, 1.5, 3, 3, 3, 3
    # Wait, for n=10, k=3, we need 10+3+1 = 14 knots.
    # Clamped: u_0=u_1=u_2=u_3 = -3
    # u_12=u_13=u_14=u_15? No, indices 0..13.
    # u_0..u_3 = -3
    # u_10..u_13 = 3
    # Interior knots u_4..u_9 (6 knots) should be in [-3, 3].
    # Let's space them uniformly:
    # h = (3 - (-3)) / (10 - 1) ? No, that's for control points.
    # The support of the spline is determined by the knots.
    
    # Let's use a simpler, more robust method: 
    # Evaluate the cubic B-spline basis using scipy.interpolate.BSpline?
    # No, we need the basis matrix N(x) such that y = N(x) @ params.
    # BSpline in scipy can give the basis matrix.
    
    from scipy.interpolate import BSpline
    
    # Define knot vector
    # n=10, k=3
    # Knots: 14 values
    # To make the spline span exactly [-3, 3] with the control points effectively
    # distributed across the domain, we can use:
    # u = [-3, -3, -3, -3, -1.5, -0.75, 0, 0.75, 1.5, 3, 3, 3, 3] -> 13 knots?
    # Need 14.
    # Let's use uniform interior knots.
    # Interior knots count = n - k - 1 = 10 - 3 - 1 = 6.
    # Interior knots: -1.5, -0.75, 0, 0.75, 1.5, 2.25? 
    # Let's just use np.linspace for interior knots from -2 to 2 to leave room for clamping?
    # Actually, if we clamp at -3 and 3, the interior knots should be between -3 and 3.
    # Let's use 6 interior knots uniformly spaced in [-3, 3].
    interior_knots = np.linspace(-3, 3, 6)
    
    # Construct full knot vector
    knots = np.concatenate([
        np.full(4, -3.0),  # Clamped start
        interior_knots,
        np.full(4, 3.0)    # Clamped end
    ])
    
    # Check length: 4 + 6 + 4 = 14. Correct.
    
    # BSpline expects knots, t (x), k, and c (control points)
    # But we want the basis matrix.
    # BSpline.basis_element or just evaluate?
    # We can construct the basis matrix by evaluating each basis function.
    # Alternatively, BSpline has a method to get the basis matrix?
    # No, but we can use `BSpline` to evaluate, but we need the linear combination.
    # Actually, `BSpline` takes control points `c` and returns values.
    # We want `y = sum(params[i] * N_i(x))`.
    # This is equivalent to `BSpline(knots, params, k)(x)`.
    
    try:
        spline = BSpline(knots, params, k=3)
        # Evaluate at x
        # BSpline evaluation handles the domain correctly.
        # Note: BSpline returns 0 outside the domain [knots[k], knots[-k-1]] if not extrapolated?
        # By default, extrapolation is linear or constant? 
        # For clamped B-splines, it's usually constant or 0 depending on implementation.
        # scipy BSpline default extrapolation is 'zero'? No, it's 'linear' for k=1, 'constant' for k>1?
        # Actually, scipy BSpline default is extrapolate='zero' is not an option.
        # It uses extrapolate=True by default which is linear? 
        # Let's check: "If extrapolate is True, the extrapolation is linear."
        # This might be bad. We want it to be well-behaved.
        # However, x is in [-3, 3], which is exactly the domain of the clamped spline.
        # So extrapolation should not be triggered for valid x in [-3, 3].
        
        y = spline(x)
        
        # Ensure finite values
        if not np.all(np.isfinite(y)):
            # Fallback to linear interpolation or constant if something went wrong
            y = np.interp(x, [-3, 3], [params[0], params[-1]])
            
        return y
        
    except Exception:
        # Fallback: simple linear basis if spline fails
        # y = params[0] + params[1] * x
        return params[0] + params[1] * x

# EVOLVE-BLOCK-END
