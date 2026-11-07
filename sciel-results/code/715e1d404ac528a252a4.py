"""Symbolic regression with 3 inputs - trigonometric harmonic basis."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a trigonometric harmonic basis.
    
    This representation is structurally different from polynomial bases and
    can capture periodic or oscillatory dependencies in the hidden function.
    
    Basis functions (10 total):
    - Constant term
    - Linear terms in x, y, z (to capture mean trend)
    - sin(2x), cos(2y), sin(2z) - single-variable harmonics
    - sin(x+y), cos(y+z), sin(z+x) - cross-term harmonics
    
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
    b4 = np.sin(2.0 * x)           # single-variable harmonic
    b5 = np.cos(2.0 * y)           # single-variable harmonic
    b6 = np.sin(2.0 * z)           # single-variable harmonic
    b7 = np.sin(x + y)             # cross-term harmonic
    b8 = np.cos(y + z)             # cross-term harmonic
    b9 = np.sin(z + x)             # cross-term harmonic
    
    # Linear combination
    return p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + \
           p[4] * b4 + p[5] * b5 + p[6] * b6 + \
           p[7] * b7 + p[8] * b8 + p[9] * b9
