"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-params basis function expansion.
    params: array of length 10
    Basis: 1, x, x^2, x^3, x^4, sin(x), cos(x), tanh(x), sin(2x), cos(2x)
    This makes the problem convex in params, so BFGS converges reliably.
    """
    # Extract 10 coefficients
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Polynomial basis
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x2
    
    # Trigonometric basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    
    # Hyperbolic tangent basis
    tanh_x = np.tanh(x)
    
    # Combine all basis functions
    result = (
        c0 * np.ones_like(x) +
        c1 * x +
        c2 * x2 +
        c3 * x3 +
        c4 * x4 +
        c5 * sin_x +
        c6 * cos_x +
        c7 * tanh_x +
        c8 * sin_2x +
        c9 * cos_2x
    )
    
    return result
# EVOLVE-BLOCK-END
