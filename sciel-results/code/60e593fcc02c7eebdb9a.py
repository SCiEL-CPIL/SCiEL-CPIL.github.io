"""Symbolic regression with 1 input using a rich basis of candidate functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combine a rich set of basis functions with 10 parameters.
    The basis includes polynomials (via normalized x), trigonometric,
    exponential, and rational terms to cover a wide range of possible
    target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x to [-1, 1] for numerical stability
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    u = (x - center) / scale
    
    # Build basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear in u
    b1 = u
    
    # 3. Quadratic in u
    b2 = u ** 2
    
    # 4. Cubic in u
    b3 = u ** 3
    
    # 5. Sine term
    b4 = np.sin(np.pi * u)
    
    # 6. Cosine term
    b5 = np.cos(np.pi * u)
    
    # 7. Exponential growth/decay
    b6 = np.exp(u)
    
    # 8. Reciprocal-like term (safe)
    b7 = 1.0 / (1.0 + np.abs(u))
    
    # 9. x^2 term (original scale)
    b8 = x ** 2
    
    # 10. Logarithmic term
    b9 = np.log1p(x)  # log(1+x), safe for x in [0.1, 6]
    
    # Stack basis functions as columns
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination with parameters
    return basis @ params
# EVOLVE-BLOCK-END
