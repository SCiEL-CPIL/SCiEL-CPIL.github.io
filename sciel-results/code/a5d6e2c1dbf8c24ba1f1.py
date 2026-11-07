"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture.
    Combines Expected Improvement (EI) with a small exploration constant (xi) to balance
    exploitation and exploration. The xi is scaled by the range of mu to adapt to the
    current posterior.
    """
    # Safety: ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute a small exploration constant xi
    # Scale by the spread of mu to adapt to the problem scale
    mu_range = np.ptp(mu)
    if mu_range < 1e-12:
        xi = 0.0
    else:
        xi = 0.01 * mu_range
    
    # Expected Improvement with exploration constant xi
    # z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma_safe
    
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip to avoid numerical issues and ensure non-negative
    ei = np.clip(ei, 0, 1e6)
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
