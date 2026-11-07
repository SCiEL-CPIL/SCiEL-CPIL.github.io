"""Sigmoid basis approximation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of shifted sigmoid basis functions:
    y = sum_{i=0}^{4} w_i / (1 + exp(-s * (x - c_i)))
    
    params layout (length 10):
      params[0..4] = weights w_0..w_4
      params[5..9] = centers c_0..c_4
    
    Fixed steepness s=3 for smoothness and BFGS convergence.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    w0, w1, w2, w3, w4 = params[0], params[1], params[2], params[3], params[4]
    c0, c1, c2, c3, c4 = params[5], params[6], params[7], params[8], params[9]
    
    # Fixed steepness
    s = 3.0
    
    # Compute sigmoids with numerical stability
    # Use the stable sigmoid: for z > 0: 1/(1+exp(-z)), for z <= 0: exp(z)/(1+exp(z))
    def stable_sigmoid(z):
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        # Use piecewise for numerical stability
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out
    
    z0 = s * (x - c0)
    z1 = s * (x - c1)
    z2 = s * (x - c2)
    z3 = s * (x - c3)
    z4 = s * (x - c4)
    
    sig0 = stable_sigmoid(z0)
    sig1 = stable_sigmoid(z1)
    sig2 = stable_sigmoid(z2)
    sig3 = stable_sigmoid(z3)
    sig4 = stable_sigmoid(z4)
    
    result = w0 * sig0 + w1 * sig1 + w2 * sig2 + w3 * sig3 + w4 * sig4
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
