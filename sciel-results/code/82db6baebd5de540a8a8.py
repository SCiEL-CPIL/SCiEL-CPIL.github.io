"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use the first 10 params to form a small, expressive basis.
    # Inputs are in [0.5, 2.0]. We build a polynomial basis in x, y, z up to
    # interactions that can be represented with 10 coefficients.
    #
    # Basis:
    # 0: 1 (intercept)
    # 1: x
    # 2: y
    # 3: z
    # 4: x*y
    # 5: x*z
    # 6: y*z
    # 7: x^2
    # 8: y^2
    # 9: z^2
    #
    # This gives a quadratic model with all pairwise interactions and squared terms,
    # which is a natural next step beyond linear. BFGS should be able to fit this
    # reasonably well for smooth functions.
    
    p = params
    n = len(x)
    
    # Ensure outputs are finite
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        pred = (p[0] + 
                p[1] * x + 
                p[2] * y + 
                p[3] * z + 
                p[4] * x * y + 
                p[5] * x * z + 
                p[6] * y * z + 
                p[7] * x * x + 
                p[8] * y * y + 
                p[9] * z * z)
    
    # Replace non-finite values with a large finite number to avoid breaking the optimizer
    pred = np.where(np.isfinite(pred), pred, 1e10)
    
    return pred
# EVOLVE-BLOCK-END
