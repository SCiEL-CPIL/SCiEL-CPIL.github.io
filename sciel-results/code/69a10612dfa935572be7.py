"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Legendre polynomials up to degree 3
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t**2 - 1)
    # P_3(t) = 0.5 * (5*t**3 - 3*t)
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 0.5 * (3.0 * x**2 - 1.0)
    px3 = 0.5 * (5.0 * x**3 - 3.0 * x)
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 0.5 * (3.0 * y**2 - 1.0)
    py3 = 0.5 * (5.0 * y**3 - 3.0 * y)
    
    # Construct 10 basis functions spanning the space of bivariate polynomials of degree <= 3
    # Order:
    # 0: P0(x)P0(y) -> 1
    # 1: P1(x)P0(y) -> x
    # 2: P0(x)P1(y) -> y
    # 3: P2(x)P0(y) -> x^2
    # 4: P1(x)P1(y) -> xy
    # 5: P0(x)P2(y) -> y^2
    # 6: P3(x)P0(y) -> x^3
    # 7: P2(x)P1(y) -> x^2y
    # 8: P1(x)P2(y) -> xy^2
    # 9: P0(x)P3(y) -> y^3
    
    pred = (
        params[0] * px0 * py0 +
        params[1] * px1 * py0 +
        params[2] * px0 * py1 +
        params[3] * px2 * py0 +
        params[4] * px1 * py1 +
        params[5] * px0 * py2 +
        params[6] * px3 * py0 +
        params[7] * px2 * py1 +
        params[8] * px1 * py2 +
        params[9] * px0 * py3
    )
    
    return pred
# EVOLVE-BLOCK-END
