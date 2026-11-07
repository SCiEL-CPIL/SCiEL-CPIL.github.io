"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed Quadratic-Cubic Polynomial Basis with [0,1] normalization.
    
    Uses inputs normalized to [0, 1] so that optimal coefficients are more likely
    to be near 1 (the BFGS starting point). Includes quadratic terms plus selected
    cubic interaction terms for additional expressiveness.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Normalize inputs to [0, 1]: range is [0.5, 2.0], width is 1.5
    x_n = (x - 0.5) / 1.5
    y_n = (y - 0.5) / 1.5
    z_n = (z - 0.5) / 1.5
    
    # 10 basis functions: constant, 3 linear, 3 squared, 3 cubic interactions
    b0 = np.ones_like(x_n)
    b1 = x_n
    b2 = y_n
    b3 = z_n
    b4 = x_n * x_n
    b5 = y_n * y_n
    b6 = z_n * z_n
    b7 = x_n * y_n * z_n
    b8 = x_n * x_n * y_n
    b9 = y_n * y_n * z_n
    
    # Compute prediction
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
