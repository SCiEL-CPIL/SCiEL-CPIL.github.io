# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative parametric form: product of two affine linear terms.
    
    pred = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7) + p8*x + p9
    
    This form can represent:
    - Products of linear combinations (e.g., (x+y)*(x-z))
    - Separable structures
    - Quadratic forms with cross terms
    - When one factor is near-constant, reduces to linear
    - Can approximate rational-like behavior through parameter scaling
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Two affine linear terms
    term1 = p0 * x + p1 * y + p2 * z + p3
    term2 = p4 * x + p5 * y + p6 * z + p7
    
    # Product plus residual linear correction
    pred = term1 * term2 + p8 * x + p9
    
    return pred
# EVOLVE-BLOCK-END
