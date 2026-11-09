"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 basis functions.
    
    Basis functions:
    1. 1
    2. x
    3. x^2
    4. x^3
    5. x^4
    6. x^5
    7. sin(x)
    8. cos(x)
    9. exp(-x^2)
    10. tanh(x)
    
    This linear model is convex with respect to parameters, allowing BFGS to find
    the global minimum efficiently. The basis spans polynomials, periodic functions,
    Gaussian-like decays, and saturating behaviors.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Pre-compute powers and functions for efficiency
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_neg_x2 = np.exp(-x2)
    tanh_x = np.tanh(x)
    
    # Construct the basis matrix (N x 10)
    # Using column_stack is efficient for small number of columns
    basis = np.column_stack((
        np.ones_like(x),
        x,
        x2,
        x3,
        x4,
        x5,
        sin_x,
        cos_x,
        exp_neg_x2,
        tanh_x
    ))
    
    # Linear combination
    return basis @ p
# EVOLVE-BLOCK-END
