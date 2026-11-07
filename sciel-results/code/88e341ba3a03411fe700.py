# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear using first two params
        p = np.asarray(params, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        if p.shape[0] >= 2:
            return p[0] * x_arr + p[1]
        return np.zeros_like(x_arr)
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Strategy: Parameterized Gaussian Mixture Model (GMM)
    # We use 3 Gaussian components.
    # The 10 parameters are distributed as:
    # p[0..2]:  Centers (mu1, mu2, mu3)
    # p[3..5]:  Widths (sigma1, sigma2, sigma3) - use softplus to ensure positivity
    # p[6..8]:  Amplitudes (a1, a2, a3)
    # p[9]:     Bias term (offset)
    
    # Extract parameters
    mu1, mu2, mu3 = p[0], p[1], p[2]
    
    # Softplus for widths to ensure they are positive and avoid division by zero
    # softplus(x) = log(1 + exp(x))
    # For stability, we can use a simpler approximation or clip
    # Since BFGS handles smooth functions well, softplus is good.
    # However, softplus can be numerically unstable for large x.
    # Let's use np.logaddexp(0, x) which is stable.
    sigma1 = np.logaddexp(0, p[3])
    sigma2 = np.logaddexp(0, p[4])
    sigma3 = np.logaddexp(0, p[5])
    
    # Ensure minimum width to prevent numerical overflow in exp
    min_sigma = 1e-3
    sigma1 = np.maximum(sigma1, min_sigma)
    sigma2 = np.maximum(sigma2, min_sigma)
    sigma3 = np.maximum(sigma3, min_sigma)
    
    a1, a2, a3 = p[6], p[7], p[8]
    bias = p[9]
    
    # Compute Gaussian components
    # g(x) = a * exp(-0.5 * ((x - mu) / sigma)^2)
    
    # To avoid overflow in exp, we can compute the exponent carefully
    # The maximum value of the exponent is 0 (when x == mu).
    # So exp is bounded by 1. This is safe.
    
    diff1 = x - mu1
    diff2 = x - mu2
    diff3 = x - mu3
    
    g1 = a1 * np.exp(-0.5 * (diff1 / sigma1)**2)
    g2 = a2 * np.exp(-0.5 * (diff2 / sigma2)**2)
    g3 = a3 * np.exp(-0.5 * (diff3 / sigma3)**2)
    
    result = g1 + g2 + g3 + bias
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs
        return p[0] * x + p[1]
    
    return result
# EVOLVE-BLOCK-END
