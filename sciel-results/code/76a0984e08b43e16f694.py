"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with 10 features covering a broad basis.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build 10 features that span a wide range of functional forms
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = x * y * z
    f8 = x * x
    f9 = y * y
    
    pred = (params[0] * f0 + 
            params[1] * f1 + 
            params[2] * f2 + 
            params[3] * f3 + 
            params[4] * f4 + 
            params[5] * f5 + 
            params[6] * f6 + 
            params[7] * f7 + 
            params[8] * f8 + 
            params[9] * f9)
    
    return pred
# EVOLVE-BLOCK-END
