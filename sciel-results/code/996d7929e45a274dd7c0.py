# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Multi-scale Fourier series expansion.
    Uses a rich basis of sine and cosine terms with multiple frequencies and interactions.
    This targets periodic or oscillatory structures that polynomials approximate poorly.
    
    Parameters (10 total):
    - p[0]: global constant offset
    - p[1..p[4]]: coefficients for first-order terms: sin(x), cos(x), sin(y), cos(y)
    - p[5..p[8]]: coefficients for second-order/interaction terms: sin(x+y), cos(x-y), sin(x*y), cos(x*y)
    - p[9]: scale factor for a higher-frequency term: sin(3*x + 3*y) or similar
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # First-order Fourier terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    
    # Interaction and higher-order terms
    sin_x_y = np.sin(x + y)
    cos_x_y = np.cos(x - y)
    sin_xy = np.sin(x * y)
    cos_xy = np.cos(x * y)
    
    # Higher frequency term
    sin_3x_3y = np.sin(3 * (x + y))
    
    # Linear combination of all basis functions
    result = (p[0] + 
              p[1] * sin_x + 
              p[2] * cos_x + 
              p[3] * sin_y + 
              p[4] * cos_y +
              p[5] * sin_x_y + 
              p[6] * cos_x_y + 
              p[7] * sin_xy + 
              p[8] * cos_xy +
              p[9] * sin_3x_3y)
    
    return result
# EVOLVE-BLOCK-END
