"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a richer basis: linear terms, quadratic terms, and interactions
    # params[0..11] correspond to:
    # 1, x, y, z, x^2, y^2, z^2, xy, xz, yz, xyz, x/y (safe division)
    # But we only have 10 params. Let's pick the most likely useful ones for a smooth function on [0.5, 2]^3.
    # Common hidden functions in symbolic regression tasks are polynomials or simple rational/exponential forms.
    # A good general-purpose basis with 10 params:
    # p0 * 1 + p1 * x + p2 * y + p3 * z + p4 * x^2 + p5 * y^2 + p6 * z^2 + p7 * xy + p8 * xz + p9 * yz
    
    p = params
    term0 = p[0]
    term1 = p[1] * x
    term2 = p[2] * y
    term3 = p[3] * z
    term4 = p[4] * x**2
    term5 = p[5] * y**2
    term6 = p[6] * z**2
    term7 = p[7] * x * y
    term8 = p[8] * x * z
    term9 = p[9] * y * z
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
