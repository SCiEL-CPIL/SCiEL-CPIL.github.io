"""Symbolic regression with 1 input using a well-conditioned basis of candidate functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combine a rich set of well-conditioned basis functions with 10 parameters.
    The basis includes Chebyshev polynomials (for smooth polynomial behavior),
    trigonometric, exponential, logarithmic, and reciprocal terms to cover
    a wide range of possible target functions.
    
    Chebyshev polynomials are used for the polynomial component because they
    are orthogonal on [-1, 1], leading to much better numerical conditioning
    for BFGS optimization compared to monomials.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Clip to valid domain to avoid issues at boundaries
    x_safe = np.clip(x, 0.1, 6.0)
    
    # Normalize x to [-1, 1] for numerical stability
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    u = (x_safe - center) / scale
    
    # Build basis functions
    # 1. Constant
    b0 = np.ones_like(x_safe)
    
    # 2-5. Chebyshev polynomials T_1 to T_4 (well-conditioned polynomial basis)
    # T_0(u) = 1 (already b0)
    # T_1(u) = u
    # T_2(u) = 2u^2 - 1
    # T_3(u) = 4u^3 - 3u
    # T_4(u) = 8u^4 - 8u^2 + 1
    b1 = u
    b2 = 2.0 * u * u - 1.0
    b3 = 4.0 * u * u * u - 3.0 * u
    b4 = 8.0 * u * u * u * u - 8.0 * u * u + 1.0
    
    # 6. Sine term (periodic behavior)
    b5 = np.sin(np.pi * u)
    
    # 7. Cosine term (periodic behavior)
    b6 = np.cos(np.pi * u)
    
    # 8. Exponential growth/decay
    b7 = np.exp(u)
    
    # 9. Logarithmic term
    b8 = np.log1p(x_safe)
    
    # 10. Reciprocal-like term (safe)
    b9 = 1.0 / (1.0 + np.abs(u))
    
    # Stack basis functions as columns
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination with parameters
    return basis @ params
# EVOLVE-BLOCK-END
