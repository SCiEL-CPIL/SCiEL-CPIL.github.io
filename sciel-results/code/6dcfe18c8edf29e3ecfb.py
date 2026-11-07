"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a combination of Expected Improvement and UCB with adaptive exploration.
    """
    # Safety: ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values in sigma by clamping
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component
    # Adaptive beta: start with moderate exploration, decay slightly
    # Use a fixed beta that balances exploitation and exploration
    beta = 1.5
    
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to have similar scales
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(ucb)
    if np.abs(ucb_max) > 1e-12:
        ucb_norm = ucb / np.abs(ucb_max)
    else:
        ucb_norm = ucb
    
    # Weighted combination
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
