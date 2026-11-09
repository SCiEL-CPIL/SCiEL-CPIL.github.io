"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute Chebyshev polynomials of the first kind up to degree 3
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct 10 basis features (tensor products)
    # Order chosen to prioritize low-degree terms
    features = np.column_stack((
        T0_x * T0_y,  # 1
        T1_x * T0_y,  # x
        T0_x * T1_y,  # y
        T2_x * T0_y,  # 2x^2 - 1
        T0_x * T2_y,  # 2y^2 - 1
        T1_x * T1_y,  # xy
        T3_x * T0_y,  # 4x^3 - 3x
        T0_x * T3_y,  # 4y^3 - 3y
        T2_x * T1_y,  # (2x^2 - 1)y
        T1_x * T2_y   # x(2y^2 - 1)
    ))
    
    # Linear combination with params
    return features @ params
# EVOLVE-BLOCK-END
