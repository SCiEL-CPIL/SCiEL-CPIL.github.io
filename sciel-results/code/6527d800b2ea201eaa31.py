"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 basis functions.
    
    Basis functions chosen to span a wide variety of common mathematical forms:
    - Polynomials (degree 0-4)
    - Trigonometric (sin(x), cos(x))
    - Gaussian decay (exp(-x^2))
    - Saturating non-linearities (tanh(x), x*tanh(x))
    
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
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_neg_x2 = np.exp(-x2)
    tanh_x = np.tanh(x)
    x_tanh_x = x * tanh_x
    
    # Construct the basis matrix (N x 10)
    # Using column_stack is efficient for small number of columns
    basis = np.column_stack((
        np.ones_like(x),      # constant
        x,                    # linear
        x2,                   # quadratic
        x3,                   # cubic
        x4,                   # quartic
        sin_x,                # sine
        cos_x,                # cosine
        exp_neg_x2,           # Gaussian
        tanh_x,               # tanh
        x_tanh_x              # x * tanh(x)
    ))
    
    # Linear combination
    result = basis @ p
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
