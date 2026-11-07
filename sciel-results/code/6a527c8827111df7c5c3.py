"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative exponential-polynomial model with 10 parameters.
    
    f(x,y,z) = p0 * exp(p1*x + p2*y + p3*z) * x^p4 * y^p5 * z^p6 
               * exp(p7*x*y + p8*y*z + p9*z*x)
    
    This captures power-law, exponential, and pairwise interaction effects
    in a single multiplicative form, structurally different from additive bases.
    """
    p = params
    
    # Clip parameters to avoid numerical overflow in exp
    p1 = np.clip(p[1], -10.0, 10.0)
    p2 = np.clip(p[2], -10.0, 10.0)
    p3 = np.clip(p[3], -10.0, 10.0)
    p4 = np.clip(p[4], -5.0, 5.0)
    p5 = np.clip(p[5], -5.0, 5.0)
    p6 = np.clip(p[6], -5.0, 5.0)
    p7 = np.clip(p[7], -5.0, 5.0)
    p8 = np.clip(p[8], -5.0, 5.0)
    p9 = np.clip(p[9], -5.0, 5.0)
    
    # Log-space computation for numerical stability
    # log(f) = log(p0) + (p1*x + p2*y + p3*z) + p4*log(x) + p5*log(y) + p6*log(z) + (p7*x*y + p8*y*z + p9*z*x)
    
    # Handle p0 sign: use sign * exp(log|p0| + rest)
    sign = np.sign(p[0])
    if sign == 0:
        sign = 1.0
    log_p0 = np.log(np.abs(p[0]) + 1e-10)
    
    log_val = (log_p0 + 
               p1 * x + p2 * y + p3 * z + 
               p4 * np.log(x) + p5 * np.log(y) + p6 * np.log(z) + 
               p7 * x * y + p8 * y * z + p9 * z * x)
    
    # Clip log_val to prevent overflow
    log_val = np.clip(log_val, -50.0, 50.0)
    
    result = sign * np.exp(log_val)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
