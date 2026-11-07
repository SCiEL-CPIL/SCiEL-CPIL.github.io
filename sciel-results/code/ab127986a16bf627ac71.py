"""Symbolic regression using shifted sigmoid basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of shifted sigmoid basis functions:
    y = p0 + p1 * sigmoid(s1*(x - c1)) + p2 * sigmoid(s2*(x - c2))
      + p3 * sigmoid(s3*(x - c3)) + p4 * sigmoid(s4*(x - c4))
      + p5 * sigmoid(s5*(x - c5))
    
    where sigmoid(z) = 1 / (1 + exp(-z))
    
    This representation is good for modeling step-like, saturating, or
    piecewise-constant behaviors while maintaining global smoothness for
    BFGS convergence.
    
    params layout (length 10):
      params[0] = constant offset p0
      params[1..5] = amplitudes p1..p5
      params[6..9] = centers c1..c4 (for 4 sigmoid terms)
    
    Wait, that's only 9 params. Let me restructure:
      params[0] = constant offset
      params[1], params[2] = amplitude and center for sigmoid 1
      params[3], params[4] = amplitude and center for sigmoid 2
      params[5], params[6] = amplitude and center for sigmoid 3
      params[7], params[8] = amplitude and center for sigmoid 4
      params[9] = sharpness scale (shared)
    
    Actually, let me use 5 sigmoids with shared sharpness:
      params[0] = constant offset
      params[1..5] = amplitudes for 5 sigmoids
      params[6..9] = centers for 4 sigmoids (only 4 fit)
    
    Let me use 4 sigmoids:
      params[0] = constant offset
      params[1..4] = amplitudes for 4 sigmoids
      params[5..8] = centers for 4 sigmoids
      params[9] = sharpness (shared scale factor)
    
    y = p0 + sum_{i=1}^{4} p_i * sigmoid(params[9] * (x - c_i))
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Extract parameters
    p0 = params[0]
    amplitudes = params[1:5]   # 4 amplitudes
    centers = params[5:9]      # 4 centers
    sharpness = params[9]
    
    # Compute sum of shifted sigmoids
    # Use a numerically stable sigmoid computation
    result = np.full_like(x, p0, dtype=float)
    
    for i in range(4):
        z = sharpness * (x - centers[i])
        # Stable sigmoid: for z >= 0, 1/(1+exp(-z)); for z < 0, exp(z)/(1+exp(z))
        # Use np.where for vectorized stable sigmoid
        sigmoid_vals = np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
            np.exp(np.clip(z, -500, 500)) / (1.0 + np.exp(np.clip(z, -500, 500)))
        )
        result = result + amplitudes[i] * sigmoid_vals
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
