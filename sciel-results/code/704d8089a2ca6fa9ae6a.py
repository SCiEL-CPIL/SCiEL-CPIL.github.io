"""Symbolic regression with 3 inputs using a mixed basis of linear, product, and smooth inverse terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a mixed basis:
    - Linear: x, y, z
    - Pairwise products: xy, xz, yz
    - Triple product: xyz
    - Smooth inverses: 1/(x+1), 1/(y+1), 1/(z+1)
    
    Total: 3 + 3 + 1 + 3 = 10 parameters
    """
    p = params
    
    # Ensure inputs are finite
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        return np.zeros_like(x, dtype=np.float64)
    
    # Compute basis terms
    # Linear
    bx = x
    by = y
    bz = z
    
    # Pairwise products
    bxy = x * y
    bxz = x * z
    byz = y * z
    
    # Triple product
    bxyz = x * y * z
    
    # Smooth inverses (well-behaved on [0.5, 2])
    b1x = 1.0 / (x + 1.0)
    b1y = 1.0 / (y + 1.0)
    b1z = 1.0 / (z + 1.0)
    
    # Combine
    result = (p[0] * bx + p[1] * by + p[2] * bz +
              p[3] * bxy + p[4] * bxz + p[5] * byz +
              p[6] * bxyz +
              p[7] * b1x + p[8] * b1y + p[9] * b1z)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x, dtype=np.float64)
    
    return result
# EVOLVE-BLOCK-END
