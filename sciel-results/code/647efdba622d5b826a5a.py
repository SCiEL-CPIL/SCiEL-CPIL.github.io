"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression equation using a diverse set of 10 basis functions.
    Combines polynomials, trigonometric, exponential, and rational terms
    to give BFGS a rich hypothesis space for fitting smooth functions on [-3,3].
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions:
    # 0: constant
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: sin(x)
    # 5: cos(x)
    # 6: exp(-x^2/2)  (Gaussian-like bump)
    # 7: exp(x/3)     (exponential growth)
    # 8: x^2/(1+x^2)  (rational, bounded)
    # 9: tanh(x)      (smooth step)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.exp(-0.5 * x ** 2)
    b7 = np.exp(x / 3.0)
    b8 = x ** 2 / (1.0 + x ** 2)
    b9 = np.tanh(x)
    
    # Clip parameters to avoid overflow in exp or large values
    p = np.clip(params, -1e4, 1e4)
    
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.clip(p[0], -1e4, 1e4) + np.clip(p[1], -1e4, 1e4) * x
    
    return result
# EVOLVE-BLOCK-END
