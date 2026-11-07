"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Polynomial basis up to degree 3
    # Features:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: xy
    # 6: x^3
    # 7: y^3
    # 8: x^2*y
    # 9: x*y^2
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    return (
        p0
        + p1 * x
        + p2 * y
        + p3 * (x ** 2)
        + p4 * (y ** 2)
        + p5 * (x * y)
        + p6 * (x ** 3)
        + p7 * (y ** 3)
        + p8 * (x ** 2 * y)
        + p9 * (x * y ** 2)
    )
# EVOLVE-BLOCK-END
