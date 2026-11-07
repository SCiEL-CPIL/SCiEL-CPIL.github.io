# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of basis functions.
    
    This version uses a Full Quadratic Polynomial basis:
    0: 1
    1: x
    2: y
    3: z
    4: x^2
    5: y^2
    6: z^2
    7: x*y
    8: x*z
    9: y*z
    
    This basis is chosen to better capture smooth, non-rational behaviors 
    or polynomials up to degree 2, which the previous rational/linear mix 
    may have approximated poorly.
    """
    # Calculate basis terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = x * x
    b5 = y * y
    b6 = z * z
    b7 = x * y
    b8 = x * z
    b9 = y * z
    
    # Stack into a matrix (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination: pred = basis @ params
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
