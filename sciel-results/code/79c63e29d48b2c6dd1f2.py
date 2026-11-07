"""Symbolic regression: Chebyshev polynomial basis for BFGS stability and accuracy."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Chebyshev polynomials up to degree 3 in x and y.
    
    Basis functions:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T0(x)T2(y) = 2y^2 - 1
    5: T1(x)T1(y) = xy
    6: T3(x)T0(y) = 4x^3 - 3x
    7: T0(x)T3(y) = 4y^3 - 3y
    8: T2(x)T1(y) = (2x^2 - 1)y
    9: T1(x)T2(y) = x(2y^2 - 1)
    
    This basis is orthogonal on [-1, 1], leading to a well-conditioned problem for BFGS.
    """
    p = params
    
    # Pre-compute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Chebyshev Polynomials T_n(x)
    # T_0 = 1
    # T_1 = x
    # T_2 = 2x^2 - 1
    # T_3 = 4x^3 - 3x
    
    T0_x = 1.0
    T1_x = x
    T2_x = 2.0 * x2 - 1.0
    T3_x = 4.0 * x3 - 3.0 * x
    
    T0_y = 1.0
    T1_y = y
    T2_y = 2.0 * y2 - 1.0
    T3_y = 4.0 * y3 - 3.0 * y
    
    # Construct predictions as linear combination
    # Using broadcasting for scalar T0
    pred = (p[0] * T0_x * T0_y + 
            p[1] * T1_x * T0_y + 
            p[2] * T0_x * T1_y + 
            p[3] * T2_x * T0_y + 
            p[4] * T0_x * T2_y + 
            p[5] * T1_x * T1_y + 
            p[6] * T3_x * T0_y + 
            p[7] * T0_x * T3_y + 
            p[8] * T2_x * T1_y + 
            p[9] * T1_x * T2_y)
            
    # Safety check for non-finite values (should not happen with this basis)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
