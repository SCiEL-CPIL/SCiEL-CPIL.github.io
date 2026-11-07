# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of basis functions
    including constants, linear terms, reciprocals, and pairwise interactions.
    
    Basis functions (10 total):
    0: 1
    1: x
    2: y
    3: z
    4: 1/x
    5: 1/y
    6: 1/z
    7: x*y
    8: x*z
    9: y*z
    
    params:
      0-9: Weights for the 10 basis functions.
    """
    # Construct basis matrix of shape (N, 10)
    # x, y, z are shape (N,)
    
    # Create column vectors for easy broadcasting if needed, but here we just stack
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = 1.0 / x
    b5 = 1.0 / y
    b6 = 1.0 / z
    b7 = x * y
    b8 = x * z
    b9 = y * z
    
    # Stack into a matrix (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination: pred = basis @ params
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
