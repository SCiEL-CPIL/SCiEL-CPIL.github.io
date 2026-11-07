"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Chebyshev polynomial basis of total degree up to 3.
    This basis is orthogonal on [-1, 1], leading to a well-conditioned
    linear system which helps BFGS converge effectively.
    
    Terms:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T1(x)T1(y) = xy
    5: T0(x)T2(y) = 2y^2 - 1
    6: T3(x)T0(y) = 4x^3 - 3x
    7: T2(x)T1(y) = (2x^2 - 1)y = 2x^2y - y
    8: T1(x)T2(y) = x(2y^2 - 1) = 2xy^2 - x
    9: T0(x)T3(y) = 4y^3 - 3y
    """
    # Pre-compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Construct the basis terms
    term0 = T0_x * T0_y
    term1 = T1_x * T0_y
    term2 = T0_x * T1_y
    term3 = T2_x * T0_y
    term4 = T1_x * T1_y
    term5 = T0_x * T2_y
    term6 = T3_x * T0_y
    term7 = T2_x * T1_y
    term8 = T1_x * T2_y
    term9 = T0_x * T3_y
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
