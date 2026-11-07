"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 2D Fourier (trigonometric) basis.
    
    The model is a linear combination of 10 basis functions:
    1. 1
    2. cos(pi * x)
    3. cos(pi * y)
    4. sin(pi * x)
    5. sin(pi * y)
    6. cos(pi * x) * cos(pi * y)
    7. cos(pi * x) * sin(pi * y)
    8. sin(pi * x) * cos(pi * y)
    9. sin(pi * x) * sin(pi * y)
    10. cos(2*pi * x) + cos(2*pi * y)
    
    This approach is distinct from polynomial regression and is well-suited for
    functions on the bounded domain [-1, 1].
    """
    p = params.astype(float)
    
    # Precompute trigonometric terms
    pi_x = np.pi * x
    pi_y = np.pi * y
    
    cos_x = np.cos(pi_x)
    sin_x = np.sin(pi_x)
    cos_y = np.cos(pi_y)
    sin_y = np.sin(pi_y)
    
    # Higher frequency term
    cos_2x = np.cos(2 * np.pi * x)
    cos_2y = np.cos(2 * np.pi * y)
    
    # Construct the linear combination
    result = (p[0] 
              + p[1] * cos_x 
              + p[2] * cos_y 
              + p[3] * sin_x 
              + p[4] * sin_y 
              + p[5] * cos_x * cos_y 
              + p[6] * cos_x * sin_y 
              + p[7] * sin_x * cos_y 
              + p[8] * sin_x * sin_y 
              + p[9] * (cos_2x + cos_2y))
              
    return result
# EVOLVE-BLOCK-END
