"""Symbolic regression with 3 inputs - rich polynomial basis approach."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a linear combination of 10 basis functions.
    
    The basis functions are designed to capture common smooth function structures:
    - Constant term
    - Linear terms in x, y, z
    - Quadratic terms (x^2, y^2, z^2)
    - Interaction terms (xy, xz, yz)
    - A cubic-like term to capture higher-order behavior
    
    params: array of 10 weights corresponding to each basis function.
    """
    p = params.flatten()
    
    # Ensure p has exactly 10 elements; if not, pad or truncate
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), mode='constant')
    else:
        p = p[:10]
    
    # Basis functions
    b0 = np.ones_like(x)          # constant
    b1 = x                         # linear x
    b2 = y                         # linear y
    b3 = z                         # linear z
    b4 = x * y                     # interaction xy
    b5 = x * z                     # interaction xz
    b6 = y * z                     # interaction yz
    b7 = x * x                     # quadratic x
    b8 = y * y                     # quadratic y
    b9 = z * z                     # quadratic z
    
    # Linear combination
    return p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + \
           p[4] * b4 + p[5] * b5 + p[6] * b6 + \
           p[7] * b7 + p[8] * b8 + p[9] * b9
