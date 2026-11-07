"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a flexible basis expansion.
    
    We use a combination of polynomial terms up to degree 5, plus logarithmic,
    exponential, and trigonometric terms to capture a wide variety of functional forms.
    
    params (length 10) are linear coefficients for the basis functions.
    The basis functions are:
        1, x, x^2, x^3, x^4, x^5, log(x), exp(-x), sin(x), cos(x)
    
    This linear-in-parameters model allows BFGS to efficiently find the best
    linear combination. Since the model is linear in params, BFGS should converge
    quickly to the global optimum for the given basis.
    """
    x = np.asarray(x, dtype=np.float64)
    # Ensure x is positive for log
    x_safe = np.maximum(x, 1e-8)
    
    # Build basis matrix
    b1 = np.ones_like(x_safe)
    b2 = x_safe
    b3 = x_safe ** 2
    b4 = x_safe ** 3
    b5 = x_safe ** 4
    b6 = x_safe ** 5
    b7 = np.log(x_safe)
    b8 = np.exp(-x_safe)
    b9 = np.sin(x_safe)
    b10 = np.cos(x_safe)
    
    # Linear combination
    result = (
        params[0] * b1 +
        params[1] * b2 +
        params[2] * b3 +
        params[3] * b4 +
        params[4] * b5 +
        params[5] * b6 +
        params[6] * b7 +
        params[7] * b8 +
        params[8] * b9 +
        params[9] * b10
    )
    
    return result
# EVOLVE-BLOCK-END
