"""Symbolic regression with Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: sum_{i=0}^{4} sum_{j=0}^{4} c_{i,j} * T_i(x) * T_j(y)
    where T_i are Chebyshev polynomials of the first kind.
    This gives a 5x5 = 25 term model, but we only have 10 params.
    
    Wait, the problem says params has length 10. So we need exactly 10 terms.
    Let's use a subset of Chebyshev products that span a reasonable space.
    
    With 10 parameters, we can use:
    - T_0(x)T_0(y) = 1
    - T_1(x)T_0(y) = x
    - T_0(x)T_1(y) = y
    - T_2(x)T_0(y) = 2x^2 - 1
    - T_0(x)T_2(y) = 2y^2 - 1
    - T_1(x)T_1(y) = xy
    - T_3(x)T_0(y) = 4x^3 - 3x
    - T_0(x)T_3(y) = 4y^3 - 3y
    - T_2(x)T_1(y) = (2x^2-1)y
    - T_1(x)T_2(y) = x(2y^2-1)
    
    This gives a 10-term model with well-conditioned Chebyshev basis.
    """
    # Compute Chebyshev polynomials T_0 through T_3 for x and y
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
    
    # Build the 10 basis functions
    # Order: (i,j) pairs
    b0 = T0_x * T0_y  # 1
    b1 = T1_x * T0_y  # x
    b2 = T0_x * T1_y  # y
    b3 = T2_x * T0_y  # 2x^2 - 1
    b4 = T0_x * T2_y  # 2y^2 - 1
    b5 = T1_x * T1_y  # xy
    b6 = T3_x * T0_y  # 4x^3 - 3x
    b7 = T0_x * T3_y  # 4y^3 - 3y
    b8 = T2_x * T1_y  # (2x^2-1)y
    b9 = T1_x * T2_y  # x(2y^2-1)
    
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
