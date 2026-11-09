"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a rich basis function expansion.
    
    Uses a combination of polynomial, trigonometric, and exponential terms
    to capture a wide variety of functional forms. The parameters are
    optimized via BFGS starting from all ones.
    
    Basis functions:
    - Constant: 1
    - Linear: x
    - Quadratic: x^2
    - Cubic: x^3
    - Quartic: x^4
    - Sine: sin(x)
    - Cosine: cos(x)
    - Sine(2x): sin(2x)
    - Cosine(2x): cos(2x)
    - Exponential: exp(-x^2) (Gaussian-like)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    exp_neg_x2 = np.exp(-x2)
    
    # Combine basis functions with parameters
    result = (
        p0 * np.ones_like(x) +
        p1 * x +
        p2 * x2 +
        p3 * x3 +
        p4 * x4 +
        p5 * sin_x +
        p6 * cos_x +
        p7 * sin_2x +
        p8 * cos_2x +
        p9 * exp_neg_x2
    )
    
    return result
# EVOLVE-BLOCK-END
