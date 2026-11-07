"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Expected Improvement (EI)
    with adaptive exploration via a UCB-like bonus scaled by uncertainty.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior standard deviation in objective units
    f_best: scalar, lowest observed value
    
    Returns: (2000,) scores, higher = more promising
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima while keeping EI as the primary driver
    # Scale the exploration bonus by the relative uncertainty
    sigma_mean = np.mean(sigma)
    exploration_weight = 0.1  # Small weight for exploration
    
    # Exploration bonus: higher sigma = more exploration
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI and exploration bonus
    score = ei + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
