# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with a trigonometric harmonic basis.
    
    Basis functions:
    0: 1
    1: sin(x)
    2: cos(x)
    3: sin(y)
    4: cos(y)
    5: sin(z)
    6: cos(z)
    7: sin(x+y)
    8: cos(x-y)
    9: sin(x+z)
    
    This basis is designed to capture periodic or oscillatory dependencies
    in the target function, which are orthogonal to the polynomial basis
    used previously.
    """
    p = params
    
    b0 = 1.0
    b1 = np.sin(x)
    b2 = np.cos(x)
    b3 = np.sin(y)
    b4 = np.cos(y)
    b5 = np.sin(z)
    b6 = np.cos(z)
    b7 = np.sin(x + y)
    b8 = np.cos(x - y)
    b9 = np.sin(x + z)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
