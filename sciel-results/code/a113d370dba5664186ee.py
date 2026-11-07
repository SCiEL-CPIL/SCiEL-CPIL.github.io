# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a multiplicative parametric form.
    
    The model is: (a1*x + b1*y + c1*z + d1) * (a2*x + b2*y + c2*z + d2)
    
    This 8-parameter (plus we use 2 more for a small additive correction)
    multiplicative form can exactly represent:
    - Products like x*y, x*z, y*z
    - Linear terms (by setting one factor to constant)
    - Rational-like structures when combined with the optimizer
    - Separable nonlinear structures
    
    With 10 params: 8 for the two affine factors, and 2 for an additive
    linear correction (alpha + beta * (x+y+z)/3) to capture any residual
    linear component not fully absorbed by the product.
    
    Actually, let's use a cleaner decomposition:
    Factor 1: a1*x + b1*y + c1*z + d1  (4 params)
    Factor 2: a2*x + b2*y + c2*z + d2  (4 params)
    Additive: e1 + e2 * (x*y*z)^(1/3)  (2 params) - geometric mean correction
    
    Total: 10 params.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    if params.size < 10:
        # Pad with zeros if needed
        params = np.concatenate([params, np.zeros(10 - params.size)])
    
    # Extract parameters
    a1, b1, c1, d1 = params[0], params[1], params[2], params[3]
    a2, b2, c2, d2 = params[4], params[5], params[6], params[7]
    e1, e2 = params[8], params[9]
    
    # Compute the two affine factors
    factor1 = a1 * x + b1 * y + c1 * z + d1
    factor2 = a2 * x + b2 * y + c2 * z + d2
    
    # Multiplicative product
    product = factor1 * factor2
    
    # Additive correction with geometric mean
    # Clip for safety
    x_safe = np.maximum(x, 1e-10)
    y_safe = np.maximum(y, 1e-10)
    z_safe = np.maximum(z, 1e-10)
    geo_mean = (x_safe * y_safe * z_safe) ** (1.0 / 3.0)
    
    correction = e1 + e2 * geo_mean
    
    pred = product + correction
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
