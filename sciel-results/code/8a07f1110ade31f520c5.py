"""Rewritten program using a linear combination of polynomial and trigonometric basis functions."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a linear combination of basis functions.
    
    The basis functions are:
    1. 1
    2. x
    3. y
    4. x^2
    5. y^2
    6. xy
    7. sin(x)
    8. cos(x)
    9. sin(y)
    10. cos(y)
    
    This model is linear in the parameters, allowing BFGS to converge to the global optimum
    (the least squares solution) reliably. This avoids the local minima issues of non-linear
    basis functions like neural networks or rational functions, while covering both polynomial
    and periodic function shapes.
    """
    # Calculate basis functions
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * x
    b5 = y * y
    b6 = x * y
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.sin(y)
    b10 = np.cos(y)
    
    # Construct the prediction
    pred = (
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
    
    return pred
