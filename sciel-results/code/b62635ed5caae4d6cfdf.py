"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Linear-in-Parameters Fixed-Basis RBF Network
# 
# The previous strategy allowed the external BFGS optimizer to adjust both 
# amplitudes and centers of the RBFs. This creates a highly non-convex 
# optimization landscape where BFGS often gets stuck in local minima or 
# converges slowly, leading to poor fits.
#
# New Strategy: Kernel Ridge Regression style
# We use a fixed set of Gaussian RBF centers evenly spaced across the input domain [-3, 3].
# The 10 parameters are simply the weights (amplitudes) for these fixed centers.
# This makes the model linear in parameters, so BFGS will converge to the global optimum
# (or very close to it) quickly and reliably.
#
# Basis:
# y = sum_{i=1 to 10} w_i * exp(-(x - c_i)^2 / (2 * sigma^2))
#
# Centers c_i are fixed: np.linspace(-3, 3, 10)
# Sigma is fixed to a value that balances local flexibility and smoothness.
# Let's try sigma = 0.8. This provides ~10 distinct bumps across the domain.

SIGMA = 0.8
CENTERS = np.linspace(-3.0, 3.0, 10)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Compute RBF features for all centers at once
    # x has shape (N,), CENTERS has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - CENTERS[np.newaxis, :]
    rbf_features = np.exp(-(diff**2) / (2.0 * SIGMA**2))
    
    # Linear combination
    result = rbf_features @ p
    
    return result
# EVOLVE-BLOCK-END
