"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model with 10 parameters.
    The basis functions are designed to span a wide variety of common
    mathematical forms while remaining stable for BFGS optimization
    starting from all ones.
    
    We use a combination of:
    - Constant term
    - Linear term
    - Polynomial terms up to degree 4
    - Common non-linear activations: tanh, sigmoid-like, exp(-x^2)
    - Interactions: x*tanh(x), x^2*tanh(x), x*exp(-x^2)
    
    The basis is:
      b0 = 1
      b1 = x
      b2 = x^2
      b3 = x^3
      b4 = x^4
      b5 = tanh(x)
      b6 = x * tanh(x)
      b7 = x^2 * tanh(x)
      b8 = x * exp(-x^2)
      b9 = exp(-x^2)
    """
    p = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    tanh_x = np.tanh(x)
    exp_neg_x2 = np.exp(-x2)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = tanh_x
    b6 = x * tanh_x
    b7 = x2 * tanh_x
    b8 = x * exp_neg_x2
    b9 = exp_neg_x2
    
    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
# EVOLVE-BLOCK-END
