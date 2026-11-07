# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable product basis with three affine linear factors.
    
    pred = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) * (p8 + p9*x + p10*y + p11*z)
    
    Wait, we only have 10 params. Let's use:
    pred = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) * (p8 + p9*x)
    
    Actually, let's keep it symmetric-ish but use 10 params:
    Factor 1: p0 + p1*x + p2*y + p3*z  (4 params)
    Factor 2: p4 + p5*x + p6*y + p7*z  (4 params)
    Factor 3: p8 + p9*x               (2 params) -- this breaks symmetry but uses all 10
    
    Alternatively, a more balanced approach:
    Factor 1: p0 + p1*x + p2*y + p3*z
    Factor 2: p4 + p5*x + p6*y + p7*z
    Factor 3: p8 + p9*z  (or some other simple form)
    
    Let's try a slightly different structure that might be more flexible:
    Use a product of two linear terms (like current) but add a third simple term
    that captures xyz or similar interactions.
    
    Actually, the directive says: "Use a separable product basis of the form 
    $(a_0 + a_1 x + a_2 y + a_3 z) \cdot (b_0 + b_1 x + b_2 y + b_3 z) \cdot (c_0 + c_1 x + c_2 y + c_3 z)$"
    
    But that's 12 params. We only have 10.
    
    Let's use:
    f1 = p0 + p1*x + p2*y + p3*z
    f2 = p4 + p5*x + p6*y + p7*z
    f3 = p8 + p9*x  (simpler third factor)
    
    pred = f1 * f2 * f3
    
    This captures terms up to cubic, including xyz, x^2y, etc.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params[:10]
    
    # Factor 1: affine linear in x, y, z
    f1 = p0 + p1 * x + p2 * y + p3 * z
    
    # Factor 2: affine linear in x, y, z
    f2 = p4 + p5 * x + p6 * y + p7 * z
    
    # Factor 3: simpler affine linear (only x and constant)
    f3 = p8 + p9 * x
    
    # Product of all three factors
    pred = f1 * f2 * f3
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
