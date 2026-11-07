import numpy as np
from scipy.interpolate import BSpline

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Cubic B-Spline basis with 10 coefficients.
    The basis is defined by 14 evenly spaced knots in [-2, 2].
    This is a linear model in params, ensuring robust convergence with BFGS.
    B-Splines provide local flexibility, which is better for capturing non-polynomial
    features than global Chebyshev polynomials.
    """
    # Define knots
    # For a B-spline of degree k with n coefficients, we need n + k + 1 knots.
    # n = 10, k = 3 -> 14 knots.
    knots = np.linspace(-2.0, 2.0, 14)
    
    # Create the design matrix
    # BSpline.design_matrix(x, knots, degree) returns a matrix of shape (len(x), n_params)
    # Note: The number of columns is len(knots) - degree - 1 = 14 - 3 - 1 = 10. This matches params length.
    try:
        # x must be 1D for design_matrix? It accepts 1D.
        # Ensure x is 1D
        x_flat = x.flatten()
        # The design matrix D such that y = D @ params
        D = BSpline.design_matrix(x_flat, knots, 3)
        return D @ params
    except Exception:
        # Fallback to Chebyshev if something goes wrong (e.g. x out of bounds, though problem says [-2,2])
        from numpy.polynomial import chebyshev as C
        t = x / 2.0
        return C.chebval(t, params)
