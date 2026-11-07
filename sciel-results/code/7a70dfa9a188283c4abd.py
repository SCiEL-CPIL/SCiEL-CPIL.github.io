"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis expansion for symbolic regression.
    
    Uses 10 parameters as coefficients for a set of basis functions.
    The basis includes polynomial, exponential, logarithmic, and trigonometric
    terms to cover a wide range of common functional forms.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: exp(x)
    5: exp(-x)
    6: log(x + 1)
    7: sin(x)
    8: cos(x)
    9: 1/x
    
    This gives BFGS a well-conditioned linear problem to solve, which should
    converge quickly and robustly for any target function that can be approximated
    by this basis on [0.1, 6].
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    # Clip x to avoid log(0) or division by zero issues, though x is in [0.1, 6]
    x_safe = np.clip(x, 1e-10, None)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = np.exp(x)
    b5 = np.exp(-x)
    b6 = np.log(x_safe + 1.0)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = 1.0 / x_safe
    
    # Combine with parameters
    p = params
    result = (p[0] * b0 +
              p[1] * b1 +
              p[2] * b2 +
              p[3] * b3 +
              p[4] * b4 +
              p[5] * b5 +
              p[6] * b6 +
              p[7] * b7 +
              p[8] * b8 +
              p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
