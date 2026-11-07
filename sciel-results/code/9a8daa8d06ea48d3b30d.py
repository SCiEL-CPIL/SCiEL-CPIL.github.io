# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a rich basis of rational, interaction,
    and polynomial terms. The evaluator fits params (length 10) via BFGS to
    minimize training MSE.
    
    Basis terms (10 total):
      1, x, y, z, 1/x, 1/y, 1/z, x*y, x*z, y*z
    
    This covers:
      - Constant and linear terms
      - Rational (reciprocal) terms for pole-like behavior
      - Pairwise interaction terms for product structures
    
    All inputs are in [0.5, 2], so reciprocals are well-behaved (range [0.5, 2]).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    
    # Extract 10 parameters
    p = params[:10]
    
    # Safe reciprocals (inputs are in [0.5, 2], so no division by zero)
    x_safe = np.where(np.abs(x) > 1e-12, x, 1e-12)
    y_safe = np.where(np.abs(y) > 1e-12, y, 1e-12)
    z_safe = np.where(np.abs(z) > 1e-12, z, 1e-12)
    
    # Build basis: 1, x, y, z, 1/x, 1/y, 1/z, x*y, x*z, y*z
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = 1.0 / x_safe
    b5 = 1.0 / y_safe
    b6 = 1.0 / z_safe
    b7 = x * y
    b8 = x * z
    b9 = y * z
    
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 +
            p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
