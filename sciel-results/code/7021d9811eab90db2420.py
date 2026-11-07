"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    
    # Basis functions
    # We limit the polynomial degree to 3 to avoid ill-conditioning
    # and use 4 non-linear terms for flexibility.
    # This reduces the parameter space from 10 to 8, helping BFGS converge.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.exp(-0.5 * x2)
    b7 = np.tanh(x)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7)
# EVOLVE-BLOCK-END
