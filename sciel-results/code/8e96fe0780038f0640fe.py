"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters symbolic regression with a rich basis of common
    mathematical functions. The model is:
    
    y = sum_{i=0}^{9} p[i] * phi_i(x)
    
    Basis functions (10 total):
    0: 1            (constant/bias)
    1: x            (linear)
    2: x^2          (quadratic)
    3: x^3          (cubic)
    4: x^4          (quartic)
    5: 1/x          (reciprocal)
    6: log(x)       (logarithmic)
    7: exp(x)       (exponential, clipped for stability)
    8: sin(x)       (sinusoidal)
    9: cos(x)       (cosinusoidal)
    
    Being linear in params makes the fitting problem convex, so BFGS
    starting from all-ones converges reliably to the global optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 0.0)
    
    # Build basis matrix
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Clip exp(x) to avoid overflow: exp(6) ~ 403, safe
    exp_x = np.exp(np.clip(x, -50, 50))
    
    # 1/x is safe since x >= 0.1
    inv_x = 1.0 / x
    
    # log(x) is safe since x >= 0.1
    log_x = np.log(x)
    
    # Compute linear combination
    prediction = (p[0] * 1.0 +
                  p[1] * x +
                  p[2] * x2 +
                  p[3] * x3 +
                  p[4] * x4 +
                  p[5] * inv_x +
                  p[6] * log_x +
                  p[7] * exp_x +
                  p[8] * np.sin(x) +
                  p[9] * np.cos(x))
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
