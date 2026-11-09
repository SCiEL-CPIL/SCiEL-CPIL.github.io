"""
Strategy: Radial Basis Function (RBF) network with fixed centers spanning the input domain.
Instead of learning center positions (which creates a non-convex landscape for BFGS),
we fix the centers to be evenly spaced across the input range [0.1, 6].
We use 8 fixed centers and 2 learnable parameters:
  - params[0]: global amplitude scale (or bias, but let's use bias + scale)
  Actually, with 10 params:
  - params[0]: bias
  - params[1]: global width parameter (shared across all RBFs, ensures smoothness and reduces params)
  - params[2:10]: 8 amplitudes for the 8 fixed centers.
  
Wait, if I share the width, I only have 1 width param. 
Let's check: 1 bias + 1 width + 8 amplitudes = 10 params.
This is a very well-conditioned problem because:
1. The basis functions are fixed (no non-convexity from center positions).
2. The width is shared, reducing the parameter space dimensionality and ensuring consistent smoothness.
3. The amplitudes are linear, so BFGS can easily adjust them.
4. The width parameter is non-linear, but it's a single parameter controlling the "sharpness" of all bumps.
   We can parameterize width as exp(params[1]) to ensure positivity, or just use params[1] directly with a softplus/abs if needed.
   Since BFGS starts at 1.0, and we want width ~ 0.5-1.0 (given 8 centers over range 5.9, spacing ~0.74), 
   using width = params[1] directly might be okay if it stays positive. But to be safe, width = exp(params[1]) is better.
   If params[1] starts at 1, width = e^1 = 2.718. That's a bit wide. 
   Maybe width = params[1] + 0.1? Or just use params[1] and hope it stays positive? 
   Let's use width = np.exp(params[1]) for stability.

Centers: evenly spaced in [0.1, 6].
8 centers: 0.1, 0.1 + 5.9/7, ..., 6.0.

This design is fundamentally different from the current GMM approach (which learns centers) 
and from Chebyshev (global basis). It is a localized basis with fixed support locations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed centers evenly spaced in [0.1, 6]
_NUM_CENTERS = 8
_DOMAIN_MIN = 0.1
_DOMAIN_MAX = 6.0
# Precompute centers
_CENTERS = np.linspace(_DOMAIN_MIN, _DOMAIN_MAX, _NUM_CENTERS)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a Radial Basis Function (RBF) network
    with fixed centers and a shared learnable width.
    params[0] = bias
    params[1] = log-width (width = exp(params[1]))
    params[2:10] = amplitudes for the 8 fixed centers
    """
    x = np.asarray(x, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)
        
    p = params
    
    bias = p[0]
    
    # Width parameter: use exp to ensure positivity
    # Clip to prevent overflow/underflow
    log_width = np.clip(p[1], -10.0, 10.0)
    width = np.exp(log_width)
    
    # Amplitudes
    amplitudes = p[2:10]
    
    # Initialize y with bias
    y = np.full_like(x, bias, dtype=float)
    
    # Compute RBF contributions
    # x shape: (N,)
    # _CENTERS shape: (8,)
    # We want to compute sum_i amplitudes[i] * exp(-0.5 * ((x - centers[i]) / width)^2)
    
    # Efficient computation using broadcasting
    # diff shape: (N, 8)
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Squared diff
    diff_sq = diff ** 2
    
    # Exponent argument
    # width is scalar
    arg = -0.5 * (diff_sq / (width ** 2))
    
    # Clip arg to avoid numerical issues (though exp handles large negatives well)
    # arg is always <= 0
    arg = np.clip(arg, -1000.0, 0.0)
    
    # Compute kernels
    kernels = np.exp(arg)  # shape (N, 8)
    
    # Weighted sum
    # amplitudes shape: (8,)
    rbf_contrib = kernels @ amplitudes  # shape (N,)
    
    y += rbf_contrib
    
    return y
# EVOLVE-BLOCK-END
