"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a bivariate trigonometric (Fourier) basis.
    This approach uses sine and cosine terms to capture periodic or oscillatory behaviors
    that polynomial bases might miss, especially for functions defined on [-1, 1].
    
    Basis terms (10 total):
    1. 1
    2. sin(pi * x)
    3. cos(pi * x)
    4. sin(pi * y)
    5. cos(pi * y)
    6. sin(2*pi * x)
    7. cos(2*pi * x)
    8. sin(2*pi * y)
    9. cos(2*pi * y)
    10. sin(pi * x) * sin(pi * y)
    
    This model is linear in parameters, ensuring BFGS converges to the global minimum.
    """
    pi = np.pi
    
    # Precompute trigonometric components
    sin_px = np.sin(pi * x)
    cos_px = np.cos(pi * x)
    sin_py = np.sin(pi * y)
    cos_py = np.cos(pi * y)
    
    sin_2px = np.sin(2 * pi * x)
    cos_2px = np.cos(2 * pi * x)
    sin_2py = np.sin(2 * pi * y)
    cos_2py = np.cos(2 * pi * y)
    
    # Interaction term
    sin_px_sin_py = sin_px * sin_py
    
    # Construct prediction
    pred = (params[0] 
            + params[1] * sin_px 
            + params[2] * cos_px 
            + params[3] * sin_py 
            + params[4] * cos_py 
            + params[5] * sin_2px 
            + params[6] * cos_2px 
            + params[7] * sin_2py 
            + params[8] * cos_2py 
            + params[9] * sin_px_sin_py)
            
    return pred
# EVOLVE-BLOCK-END
