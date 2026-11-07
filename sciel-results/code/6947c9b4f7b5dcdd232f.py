"""
Radial Basis Function (RBF) + Polynomial basis for symbolic regression.

Structure:
- 5 Gaussian RBF kernels with fixed centers at {-3, -1.5, 0, 1.5, 3}
- 5 polynomial terms: 1, x, x^2, x^3, x^4

This creates a localized representation that can capture both global trends
(polynomial) and local features (RBF bumps). The model is linear in parameters,
so BFGS solves a convex problem reliably from the all-ones starting point.

The Gaussian width sigma is chosen to provide moderate localization:
- Centers are spaced 1.5 apart
- sigma = 1.0 gives good overlap between adjacent kernels
- This allows BFGS to independently adjust local "bumps" while the
  polynomial captures the overall trend
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF + Polynomial basis function.
    
    Parameters:
        x: array of input values
        params: array of 10 parameters
            params[0:5] - coefficients for polynomial terms (1, x, x^2, x^3, x^4)
            params[5:10] - coefficients for RBF kernels at centers {-3, -1.5, 0, 1.5, 3}
    
    Returns:
        Array of predictions, one per input point
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial basis (5 terms)
    y = (params[0] * np.ones_like(x) +
         params[1] * x +
         params[2] * x**2 +
         params[3] * x**3 +
         params[4] * x**4)
    
    # RBF basis (5 Gaussian kernels)
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0  # Gaussian width
    
    for i, c in enumerate(centers):
        y += params[5 + i] * np.exp(-0.5 * ((x - c) / sigma)**2)
    
    return y
