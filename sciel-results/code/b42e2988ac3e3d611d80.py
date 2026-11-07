"""Multiplicative parametric form: product of two affine linear terms."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a multiplicative parametric form:
    
    f(x, y, z) = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7) * (p8*x + p9*y + p10*z + p11)
    
    Wait, we only have 10 params. Let's use:
    f(x, y, z) = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7)
    
    That's 8 params. We have 10. Let's add a scaling and offset:
    f(x, y, z) = p8 * (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7) + p9
    
    This is a multiplicative interaction model with a global scale and offset.
    It can represent:
    - Pure products: xy, xz, yz, x^2, y^2, z^2
    - Mixed terms with different weights
    - Rational-like structures when one factor is small
    
    The key structural difference from the additive basis is that this form
    naturally captures multiplicative interactions without needing explicit
    interaction terms in a linear basis.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure params has exactly 10 elements
    if params.size != 10:
        # Pad or truncate to 10
        if params.size < 10:
            params = np.pad(params, (0, 10 - params.size), constant_values=0.0)
        else:
            params = params[:10]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # First affine factor
    factor1 = p0 * x + p1 * y + p2 * z + p3
    
    # Second affine factor
    factor2 = p4 * x + p5 * y + p6 * z + p7
    
    # Multiply and scale/offset
    pred = p8 * factor1 * factor2 + p9
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
