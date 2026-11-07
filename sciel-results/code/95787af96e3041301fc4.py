import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis:
    - Linear terms in all variables
    - Interaction terms (xy, xz, yz) to capture cross-dependencies
    - Sine terms for periodic components
    - Cosine terms for additional periodic coverage
    
    Inputs are normalized to [-1, 1] for numerical stability.
    The 10 params are distributed across these basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Normalize inputs to [-1, 1] from [0.5, 2]
    xn = (x - 1.25) / 0.75
    yn = (y - 1.25) / 0.75
    zn = (z - 1.25) / 0.75
    
    # Basis 0: constant
    t0 = np.ones_like(x)
    
    # Basis 1: linear x
    t1 = xn
    
    # Basis 2: linear y
    t2 = yn
    
    # Basis 3: linear z
    t3 = zn
    
    # Basis 4: interaction xy
    t4 = xn * yn
    
    # Basis 5: interaction xz
    t5 = xn * zn
    
    # Basis 6: interaction yz
    t6 = yn * zn
    
    # Basis 7: sine of x
    t7 = np.sin(x)
    
    # Basis 8: sine of y
    t8 = np.sin(y)
    
    # Basis 9: sine of z
    t9 = np.sin(z)
    
    result = (p[0] * t0 + 
              p[1] * t1 + 
              p[2] * t2 + 
              p[3] * t3 + 
              p[4] * t4 + 
              p[5] * t5 + 
              p[6] * t6 + 
              p[7] * t7 + 
              p[8] * t8 + 
              p[9] * t9)
              
    return result
