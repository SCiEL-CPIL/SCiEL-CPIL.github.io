# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Strategy: Parameterized Gaussian Mixture Model (3 components)
    # Parameters:
    # p[0], p[1], p[2]: Centers (mu1, mu2, mu3)
    # p[3], p[4], p[5]: Widths (sigma1, sigma2, sigma3) - use softplus to ensure positive
    # p[6], p[7], p[8]: Heights (h1, h2, h3)
    # p[9]: Constant offset (bias)
    
    # Extract parameters
    mu1, mu2, mu3 = p[0], p[1], p[2]
    
    # Softplus for widths to ensure they are positive and non-zero, avoiding division by zero
    # sigma = log(1 + exp(alpha * param))
    # Using alpha=1.0 for reasonable sensitivity
    def softplus(z):
        return np.log1p(np.exp(z))
        
    sigma1 = softplus(p[3]) + 1e-6
    sigma2 = softplus(p[4]) + 1e-6
    sigma3 = softplus(p[5]) + 1e-6
    
    h1, h2, h3 = p[6], p[7], p[8]
    bias = p[9]
    
    # Compute Gaussians
    # g(x) = h * exp(-0.5 * ((x - mu) / sigma)^2)
    
    # Vectorized computation for efficiency
    diff1 = (x - mu1) / sigma1
    diff2 = (x - mu2) / sigma2
    diff3 = (x - mu3) / sigma3
    
    # Clip exponent to avoid underflow/overflow issues, though exp(-large) is 0
    # exp(-0.5 * diff^2)
    # If diff is large, exp is 0. If diff is small, exp is ~1.
    # Safe to compute directly for x in [-3,3] and reasonable sigmas.
    
    g1 = h1 * np.exp(-0.5 * diff1**2)
    g2 = h2 * np.exp(-0.5 * diff2**2)
    g3 = h3 * np.exp(-0.5 * diff3**2)
    
    result = g1 + g2 + g3 + bias
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
