"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a hidden function using a full cubic polynomial basis.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: x*y
    5: y^2
    6: x^3
    7: x^2*y
    8: x*y^2
    9: y^3
    
    This uses exactly 10 parameters, matching the required array length.
    The model is linear in the parameters, allowing BFGS to optimize
    the convex quadratic loss function efficiently.
    """
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Calculate basis terms
    # Using Horner-like or direct vectorization for speed
    # x and y are 1D arrays
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Construct prediction
    # params[0] * 1 + params[1] * x + ...
    pred = (params[0] + 
            params[1] * x + 
            params[2] * y + 
            params[3] * x2 + 
            params[4] * xy + 
            params[5] * y2 + 
            params[6] * x3 + 
            params[7] * x2y + 
            params[8] * xy2 + 
            params[9] * y3)
            
    return pred
# EVOLVE-BLOCK-END
