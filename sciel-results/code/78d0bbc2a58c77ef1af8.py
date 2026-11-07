"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a rich, well-conditioned basis of functions.
    
    We use 10 parameters as coefficients for a linear combination of 10 basis functions.
    The basis is chosen to cover common functional forms and is standardized
    (shifted/scaled) to improve conditioning for BFGS optimization.
    
    Basis functions (x assumed in [0.1, 6]):
    0: constant (1)
    1: linear (x_shifted)
    2: quadratic (x_shifted^2)
    3: cubic (x_shifted^3)
    4: exponential (exp(x_shifted))
    5: inverse exponential (exp(-x_shifted))
    6: logarithmic (log(x + 1))
    7: sine (sin(x))
    8: cosine (cos(x))
    9: reciprocal (1/(x + 0.1))
    """
    # Deadline guard
    t_start = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Standardize x to improve conditioning
    # x is in [0.1, 6], so shift and scale to [0, 1]
    x_shifted = (x - 0.1) / (6.0 - 0.1)
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x_shifted
    b2 = x_shifted ** 2
    b3 = x_shifted ** 3
    b4 = np.exp(x_shifted)
    b5 = np.exp(-x_shifted)
    b6 = np.log(x + 1.0)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = 1.0 / (x + 0.1)
    
    # Combine with parameters
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    # Guard: keep output finite
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
