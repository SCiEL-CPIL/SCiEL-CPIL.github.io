"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a basis of Chebyshev polynomials up to degree 3 in x and y.
    This allows fitting smooth functions including quadratics and cubics,
    which the linear seed cannot.
    """
    # Chebyshev polynomials of the first kind
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 basis functions
    # 1. Constant
    b0 = T0_x * T0_y
    # 2. Linear x
    b1 = T1_x * T0_y
    # 3. Linear y
    b2 = T0_x * T1_y
    # 4. Quadratic x
    b3 = T2_x * T0_y
    # 5. Quadratic y
    b4 = T0_x * T2_y
    # 6. Interaction xy
    b5 = T1_x * T1_y
    # 7. Cubic x
    b6 = T3_x * T0_y
    # 8. Cubic y
    b7 = T0_x * T3_y
    # 9. Mixed x^2 y
    b8 = T2_x * T1_y
    # 10. Mixed x y^2
    b9 = T1_x * T2_y
    
    # Combine with parameters
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
