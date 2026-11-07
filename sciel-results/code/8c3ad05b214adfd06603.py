"""Symbolic regression using smooth rational function approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation:
    y = (p0 + p1*t + p2*t^2 + p3*t^3 + p4*t^4) / (1 + q0*t + q1*t^2)
    
    where t = x/2 to scale to [-1, 1].
    
    params layout (length 10):
      params[0..4] = numerator coefficients p0..p4
      params[5..6] = denominator coefficients q0, q1 (denom = 1 + q0*t + q1*t^2)
      params[7..9] = unused (kept for interface compatibility)
    
    The denominator is made smooth and positive via softplus to avoid
    non-differentiable points that would confuse BFGS.
    """
    start_time = time.time()
    DEADLINE = 145.0
    
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Scale x to [-1, 1]
    t = x * 0.5
    
    # Extract parameters
    p0, p1, p2, p3, p4 = params[0], params[1], params[2], params[3], params[4]
    q0, q1 = params[5], params[6]
    
    # Compute numerator: polynomial in t
    numerator = p0 + p1*t + p2*t**2 + p3*t**3 + p4*t**4
    
    # Compute raw denominator: 1 + q0*t + q1*t^2
    raw_denom = 1.0 + q0*t + q1*t**2
    
    # Use softplus to ensure denominator is always positive and smooth
    # softplus(z) = log(1 + exp(z)), which is smooth and > 0
    # Scale: we want denom = softplus(raw_denom * scale) / scale + offset
    # For large scale, softplus(z*scale)/scale approximates max(z, 0)
    # But we want it to be close to raw_denom when raw_denom is positive
    # and positive when raw_denom is negative
    # Let's use: denom = softplus(raw_denom) + 0.1
    # This ensures denom > 0.1 always, and is smooth everywhere
    
    # Actually, let's use a simpler approach: 
    # denom = raw_denom + exp(|raw_denom|) * 0.1
    # No, that's not smooth either.
    
    # Best approach: denom = softplus(raw_denom) + eps
    # softplus is smooth, and for large positive raw_denom, softplus ≈ raw_denom
    # For large negative raw_denom, softplus ≈ 0
    # So denom is always > eps
    
    eps = 1e-6
    denom = np.logaddexp(0.0, raw_denom) + eps
    
    # Compute prediction
    result = numerator / denom
    
    # Check time
    if time.time() > DEADLINE:
        pass
        
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
# EVOLVE-BLOCK-END
