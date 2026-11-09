"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 2D Chebyshev polynomial basis.
    Chebyshev polynomials are orthogonal on [-1, 1] and bounded, 
    leading to a well-conditioned least-squares problem suitable for BFGS.
    """
    # Extract parameters
    c00 = params[0]
    c10 = params[1]
    c01 = params[2]
    c20 = params[3]
    c02 = params[4]
    c11 = params[5]
    c30 = params[6]
    c03 = params[7]
    c21 = params[8]
    c12 = params[9]
    
    # Compute Chebyshev polynomials for x and y
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the linear combination
    # Basis: T_i(x) * T_j(y)
    # 1. T0(x)T0(y)
    # 2. T1(x)T0(y)
    # 3. T0(x)T1(y)
    # 4. T2(x)T0(y)
    # 5. T0(x)T2(y)
    # 6. T1(x)T1(y)
    # 7. T3(x)T0(y)
    # 8. T0(x)T3(y)
    # 9. T2(x)T1(y)
    # 10. T1(x)T2(y)
    
    return (c00 * T0_x * T0_y +
            c10 * T1_x * T0_y +
            c01 * T0_x * T1_y +
            c20 * T2_x * T0_y +
            c02 * T0_x * T2_y +
            c11 * T1_x * T1_y +
            c30 * T3_x * T0_y +
            c03 * T0_x * T3_y +
            c21 * T2_x * T1_y +
            c12 * T1_x * T2_y)
# EVOLVE-BLOCK-END
