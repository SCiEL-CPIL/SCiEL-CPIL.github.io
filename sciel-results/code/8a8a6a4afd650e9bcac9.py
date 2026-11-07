# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining polynomial-cross terms, reciprocal terms, and ratio terms.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: x
    2: y
    3: z
    4: x*y
    5: y*z
    6: z*x
    7: 1/x
    8: 1/y
    9: 1/z
    
    This basis captures:
    - Linear dependencies
    - Pairwise multiplicative interactions (common in symbolic regression)
    - Reciprocal structure for rational-like functions
    """
    p = params
    
    b0 = 1.0
    b1 = x
    b2 = y
    b3 = z
    b4 = x * y
    b5 = y * z
    b6 = z * x
    b7 = 1.0 / x
    b8 = 1.0 / y
    b9 = 1.0 / z
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
