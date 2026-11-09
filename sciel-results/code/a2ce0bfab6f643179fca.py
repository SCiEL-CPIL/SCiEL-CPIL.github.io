"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis functions:
    p0: 1
    p1: x
    p2: y
    p3: z
    p4: xy
    p5: xz
    p6: yz
    p7: xyz
    p8: 1/x
    p9: 1/y
    """
    # Calculate features
    # Note: x, y, z are in [0.5, 2], so 1/x and 1/y are safe and finite.
    
    # To be safe against potential division by zero (though range is [0.5, 2]), 
    # we don't strictly need epsilon, but it's good practice.
    # However, the problem states inputs are from [0.5, 2].
    
    f0 = 1.0
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = x * y * z
    f8 = 1.0 / x
    f9 = 1.0 / y
    
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
