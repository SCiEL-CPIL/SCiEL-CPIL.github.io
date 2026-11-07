"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter separable tensor-product Chebyshev basis.
    Uses T_i(x) * T_j(y) for i,j in {0,1,2} plus additional terms.
    Chebyshev polynomials are well-conditioned on [-1,1] and orthogonal,
    providing a stable optimization landscape for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Chebyshev polynomials of the first kind
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x * x - 1
    T3_x = 4 * x**3 - 3 * x
    T4_x = 8 * x**4 - 8 * x**2 + 1
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y * y - 1
    T3_y = 4 * y**3 - 3 * y
    T4_y = 8 * y**4 - 8 * y**2 + 1
    
    # Tensor-product basis: T_i(x) * T_j(y) for i,j in {0,1,2}
    # This gives 9 terms, capturing rank-3 interactions
    b0 = T0_x * T0_y      # 1
    b1 = T1_x * T0_y      # x
    b2 = T0_x * T1_y      # y
    b3 = T2_x * T0_y      # 2x^2 - 1
    b4 = T0_x * T2_y      # 2y^2 - 1
    b5 = T1_x * T1_y      # x*y
    b6 = T2_x * T1_y      # (2x^2-1)*y
    b7 = T1_x * T2_y      # x*(2y^2-1)
    b8 = T2_x * T2_y      # (2x^2-1)*(2y^2-1)
    
    # Additional higher-order term for flexibility
    b9 = T3_x * T3_y      # (4x^3-3x)*(4y^3-3y)
    
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
