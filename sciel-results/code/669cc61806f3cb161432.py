"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining low-degree polynomial basis for global structure
    with Gaussian RBF terms at the four corners for local flexibility.
    
    Basis terms:
    0: 1 (bias)
    1: x
    2: y
    3: x^2
    4: y^2
    5: xy
    6: RBF at (1, 1)
    7: RBF at (-1, -1)
    8: RBF at (1, -1)
    9: RBF at (-1, 1)
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Polynomial terms for global structure
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # RBF terms at all four corners for local flexibility
    sigma = 0.7
    sigma2 = 2.0 * sigma * sigma
    
    r1_sq = (x - 1.0)**2 + (y - 1.0)**2
    r2_sq = (x + 1.0)**2 + (y + 1.0)**2
    r3_sq = (x - 1.0)**2 + (y + 1.0)**2
    r4_sq = (x + 1.0)**2 + (y - 1.0)**2
    
    g1 = np.exp(-r1_sq / sigma2)
    g2 = np.exp(-r2_sq / sigma2)
    g3 = np.exp(-r3_sq / sigma2)
    g4 = np.exp(-r4_sq / sigma2)
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * y2 +
        p[5] * xy +
        p[6] * g1 +
        p[7] * g2 +
        p[8] * g3 +
        p[9] * g4
    )
    return pred
# EVOLVE-BLOCK-END
