# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative parametric form: product of two affine linear terms in (x, y, z).
    
    pred = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7) + p8*x + p9*y
    
    This can represent:
    - Pure products: xy, xz, yz, x*y*z (with appropriate zeroing)
    - Quadratic interactions
    - Some rational-like structures when combined with the additive tail
    - Separable non-linear structures
    
    The additive tail (p8*x + p9*y) provides additional linear capacity.
    All 10 parameters are used.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # First affine factor
    a1 = p0 * x + p1 * y + p2 * z + p3
    
    # Second affine factor
    a2 = p4 * x + p5 * y + p6 * z + p7
    
    # Product of the two affine terms
    product_term = a1 * a2
    
    # Additive linear tail for additional flexibility
    linear_tail = p8 * x + p9 * y
    
    pred = product_term + linear_tail
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
