"""Symbolic regression with 3 inputs - Rich polynomial basis with cubic terms."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a full cubic polynomial basis in x, y, z.
    
    Basis (10 terms):
    1. x
    2. y
    3. z
    4. x*y
    5. y*z
    6. z*x
    7. x^2
    8. y^2
    9. z^2
    10. x*y*z
    
    This is linear in parameters, so the optimization is convex (quadratic loss),
    and BFGS will reliably find the global minimum. The cubic interaction term
    x*y*z adds expressiveness beyond the quadratic basis.
    """
    p = params.flatten()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), mode='constant')
    else:
        p = p[:10]
        
    a, b, c, d, e, f, g, h, i, j = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Linear terms
    linear = a * x + b * y + c * z
    
    # Pairwise interaction terms
    interactions = d * x * y + e * y * z + f * z * x
    
    # Squared terms
    squared = g * x * x + h * y * y + i * z * z
    
    # Cubic interaction term
    cubic = j * x * y * z
    
    pred = linear + interactions + squared + cubic
    
    return pred
