# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining polynomial, reciprocal, and ratio terms.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: x
    2: y
    3: z
    4: x*y
    5: y*z
    6: z*x
    7: 1/x
    8: x/y
    9: y/z
    
    This basis captures:
    - Linear dependencies
    - Pairwise multiplicative interactions
    - Reciprocal structure for rational-like functions
    - Ratio terms for additional rational structure
    """
    p = params.flatten()
    
    # Ensure we have at least 10 parameters
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    a4 = p[4]
    a5 = p[5]
    a6 = p[6]
    a7 = p[7]
    a8 = p[8]
    a9 = p[9]
    
    # Build the expression
    result = a0 * 1.0
    result += a1 * x
    result += a2 * y
    result += a3 * z
    result += a4 * x * y
    result += a5 * y * z
    result += a6 * z * x
    result += a7 / x
    result += a8 * x / y
    result += a9 * y / z
    
    return result
# EVOLVE-BLOCK-END
