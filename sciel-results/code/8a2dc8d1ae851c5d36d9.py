"""Trigonometric basis for symbolic regression with 2 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric basis function using 10 parameters:
    - params[0]: intercept (1)
    - params[1]: cos(pi*x)
    - params[2]: sin(pi*x)
    - params[3]: cos(pi*y)
    - params[4]: sin(pi*y)
    - params[5]: cos(pi*x)*cos(pi*y)
    - params[6]: sin(pi*x)*sin(pi*y)
    - params[7]: cos(pi*(x+y))
    - params[8]: sin(pi*(x+y))
    - params[9]: cos(pi*(x-y))
    """
    p = params
    
    # Precompute trigonometric terms
    cos_x = np.cos(np.pi * x)
    sin_x = np.sin(np.pi * x)
    cos_y = np.cos(np.pi * y)
    sin_y = np.sin(np.pi * y)
    
    cos_xy = np.cos(np.pi * (x + y))
    sin_xy = np.sin(np.pi * (x + y))
    cos_xm = np.cos(np.pi * (x - y))
    
    # Linear combination of basis functions
    pred = (
        p[0]
        + p[1] * cos_x
        + p[2] * sin_x
        + p[3] * cos_y
        + p[4] * sin_y
        + p[5] * cos_x * cos_y
        + p[6] * sin_x * sin_y
        + p[7] * cos_xy
        + p[8] * sin_xy
        + p[9] * cos_xm
    )
    
    # Replace any non-finite values with 0 to avoid optimizer issues
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
