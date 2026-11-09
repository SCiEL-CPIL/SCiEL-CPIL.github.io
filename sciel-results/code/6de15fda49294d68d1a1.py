"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses a combination of Expected Improvement (EI) and an exploration
    bonus based on uncertainty to balance exploitation and exploration.
    """
    start_time = time.time()
    deadline = 325.0
    safety_margin = 5.0

    n = len(mu)
    
    # Handle sigma ~ 0 cases to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Add exploration bonus: regions with high uncertainty get a boost
    # This helps escape local optima in the Weierstrass function
    # The bonus is scaled to be significant but not dominant
    max_sigma = np.max(sigma_safe)
    if max_sigma > 0:
        exploration_bonus = 0.1 * sigma_safe / max_sigma
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    # Combined score: EI + exploration bonus
    score = ei + exploration_bonus
    
    # Check time
    if time.time() > deadline - safety_margin:
        return score
    
    return score

# EVOLVE-BLOCK-END
