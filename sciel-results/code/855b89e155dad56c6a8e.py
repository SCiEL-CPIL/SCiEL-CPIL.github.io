# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative parametric form: product of two affine linear terms in (x, y, z).
    
    This structure can represent:
    - Products like x*y, x*z, y*z, x*y*z
    - Sums of products (expanded form)
    - Rational-like behavior when one factor is small
    - Separable structures
    
    pred = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7)
    
    This gives 8 parameters. We use the remaining 2 as scaling/offset:
    pred = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7) * p8 + p9
    
    The multiplicative structure (p8) and additive offset (p9) add flexibility.
    
    This is a nonlinear model in params, so BFGS is used. Starting from all ones,
    the initial prediction is (3x+3y+3z+1)*(3x+3y+3z+1)*1+1 which is finite
    for inputs in [0.5, 2].
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # First affine factor
    a1 = p0 * x + p1 * y + p2 * z + p3
    
    # Second affine factor
    a2 = p4 * x + p5 * y + p6 * z + p7
    
    # Multiplicative combination with scaling and offset
    pred = a1 * a2 * p8 + p9
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
