"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-of-terms basis with multiple centers.
    params: 10 parameters total.
    We use a basis of x^k and (x-c)^k for distinct centers c in [0.1, 6].
    
    Design:
    - params[0] * 1
    - params[1] * x
    - params[2] * x^2
    - params[3] * x^3
    - params[4] * (x - 1.5)^2
    - params[5] * (x - 3.0)^2
    - params[6] * (x - 4.5)^2
    - params[7] * (x - 1.5)^3
    - params[8] * (x - 3.0)^3
    - params[9] * (x - 4.5)^3
    
    This creates a flexible algebraic basis that can capture multi-scale behaviors.
    The basis is linear in params, so BFGS will converge reliably.
    """
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    
    c1 = x - 1.5
    c2 = x - 3.0
    c3 = x - 4.5
    
    c1_sq = c1 * c1
    c2_sq = c2 * c2
    c3_sq = c3 * c3
    
    c1_cu = c1_sq * c1
    c2_cu = c2_sq * c2
    c3_cu = c3_sq * c3
    
    # Build prediction as linear combination
    pred = (params[0] * 1.0 +
            params[1] * x +
            params[2] * x2 +
            params[3] * x3 +
            params[4] * c1_sq +
            params[5] * c2_sq +
            params[6] * c3_sq +
            params[7] * c1_cu +
            params[8] * c2_cu +
            params[9] * c3_cu)
    
    return pred
# EVOLVE-BLOCK-END
