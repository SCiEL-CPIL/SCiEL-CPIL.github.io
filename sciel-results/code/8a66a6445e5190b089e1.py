"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Pre-compute Chebyshev polynomials T_0..T_5
# T_0(x) = 1
# T_1(x) = x
# T_2(x) = 2x^2 - 1
# T_3(x) = 4x^3 - 3x
# T_4(x) = 8x^4 - 8x^2 + 1
# T_5(x) = 16x^5 - 20x^3 + 5x

def chebyshev_matrix(x, y):
    """
    Construct design matrix with 10 columns:
    T_0(x)T_0(y), T_1(x)T_0(y), T_0(x)T_1(y), T_2(x)T_0(y), T_0(x)T_2(y),
    T_1(x)T_1(y), T_3(x)T_0(y), T_0(x)T_3(y), T_2(x)T_1(y), T_1(x)T_2(y)
    
    This selection covers low-order interactions and is well-conditioned.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute Chebyshev polynomials for x
    T0_x = np.ones_like(x)
    T1_x = x.copy()
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    T4_x = 8 * x**4 - 8 * x**2 + 1
    T5_x = 16 * x**5 - 20 * x**3 + 5 * x
    
    # Compute Chebyshev polynomials for y
    T0_y = np.ones_like(y)
    T1_y = y.copy()
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    T4_y = 8 * y**4 - 8 * y**2 + 1
    T5_y = 16 * y**5 - 20 * y**3 + 5 * y
    
    # Build 10 feature columns
    # Selection: prioritize low-degree terms and key interactions
    # Order: constant, linear x, linear y, quadratic x, quadratic y, 
    #        interaction xy, cubic x, cubic y, x^2*y, x*y^2
    cols = [
        T0_x * T0_y,  # 1
        T1_x * T0_y,  # x
        T0_x * T1_y,  # y
        T2_x * T0_y,  # 2x^2 - 1
        T0_x * T2_y,  # 2y^2 - 1
        T1_x * T1_y,  # x*y
        T3_x * T0_y,  # 4x^3 - 3x
        T0_x * T3_y,  # 4y^3 - 3y
        T2_x * T1_y,  # (2x^2-1)*y
        T1_x * T2_y,  # x*(2y^2-1)
    ]
    
    return np.column_stack(cols)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using Chebyshev polynomial basis.
    The evaluator fits params via BFGS starting from all ones.
    """
    X = chebyshev_matrix(x, y)
    # params is shape (10,), X is shape (n, 10)
    return X @ params
# EVOLVE-BLOCK-END
