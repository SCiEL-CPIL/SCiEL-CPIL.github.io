"""Symbolic regression using rational function approximation."""
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
      params[7..9] = unused (kept for compatibility)
    
    The denominator is kept positive via soft clipping to avoid singularities
    and ensure BFGS stability.
    
    Parameters:
    x: Input array, shape (N,)
    params: Coefficients array, shape (10,)
    
    Returns:
    Predicted y values, shape (N,)
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Scale x to [-1, 1]
    t = x * 0.5
    
    # Extract parameters
    p0, p1, p2, p3, p4 = params[0], params[1], params[2], params[3], params[4]
    q0, q1 = params[5], params[6]
    
    # Compute numerator: p0 + p1*t + p2*t^2 + p3*t^3 + p4*t^4
    numerator = p0 + p1 * t + p2 * t**2 + p3 * t**3 + p4 * t**4
    
    # Compute denominator: 1 + q0*t + q1*t^2
    denominator = 1.0 + q0 * t + q1 * t**2
    
    # Soft clip denominator to ensure it stays away from zero
    # Use a large threshold to minimize distortion when denominator is well-behaved
    # If denominator becomes too small in magnitude, we clip it
    min_denom = 1e-6
    # For positive denominators, clip from below
    # For negative denominators, clip from above (towards -min_denom)
    # We want |denominator| >= min_denom
    # Use np.where to handle sign
    pos_mask = denominator >= 0
    denominator = np.where(pos_mask, 
                            np.maximum(denominator, min_denom),
                            np.minimum(denominator, -min_denom))
    
    # Compute result
    result = numerator / denominator
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
