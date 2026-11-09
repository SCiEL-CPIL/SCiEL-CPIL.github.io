"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Expected Improvement (EI) with a small exploration bonus to
    balance exploitation and exploration.
    
    Args:
        mu: array of shape (2000,) - posterior means in objective units
        sigma: array of shape (2000,) - posterior standard deviations in objective units
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher meaning evaluate next
    """
    start_time = time.time()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Add a small exploration bonus proportional to sigma
    # This encourages visiting regions with high uncertainty
    # The coefficient is small to keep EI as the primary signal
    exploration_coeff = 0.05
    scores = ei + exploration_coeff * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we're running too long, return immediately
    if time.time() - start_time > 320.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
