"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions spanning common mathematical forms.
    
    Basis functions:
    1. 1 (constant)
    2. x
    3. x^2
    4. x^3
    5. x^4
    6. sin(x)
    7. cos(x)
    8. exp(-x^2) (Gaussian)
    9. tanh(x)
    10. x * tanh(x)
    
    This is a linear model in the parameters, so BFGS will find the global
    minimum efficiently (convex optimization). The basis is chosen to span
    polynomials, periodic functions, Gaussian decays, and saturating behaviors.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Pre-compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_neg_x2 = np.exp(-x2)
    tanh_x = np.tanh(x)
    x_tanh_x = x * tanh_x
    
    # Linear combination of basis functions
    result = (
        p[0] * 1.0
        + p[1] * x
        + p[2] * x2
        + p[3] * x3
        + p[4] * x4
        + p[5] * sin_x
        + p[6] * cos_x
        + p[7] * exp_neg_x2
        + p[8] * tanh_x
        + p[9] * x_tanh_x
    )
    
    return result
# EVOLVE-BLOCK-END
