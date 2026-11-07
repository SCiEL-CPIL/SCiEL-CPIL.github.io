"""Rewritten program: hybrid basis with tanh nonlinearities, polynomial terms, and interactions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis function using 10 parameters:
    - params[0]: intercept
    - params[1]: x
    - params[2]: y
    - params[3]: x^2
    - params[4]: y^2
    - params[5]: x*y
    - params[6]: tanh(x)
    - params[7]: tanh(y)
    - params[8]: tanh(x+y)
    - params[9]: tanh(x-y)
    """
    p = params
    
    # Ensure finite output
    pred = (
        p[0]
        + p[1] * x
        + p[2] * y
        + p[3] * x**2
        + p[4] * y**2
        + p[5] * x * y
        + p[6] * np.tanh(x)
        + p[7] * np.tanh(y)
        + p[8] * np.tanh(x + y)
        + p[9] * np.tanh(x - y)
    )
    
    # Replace any non-finite values with 0 to avoid optimizer issues
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
