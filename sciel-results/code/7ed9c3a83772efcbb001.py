"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement per unit cost with an additional exploration bonus
    to balance exploitation and exploration.
    """
    # Floor sigma to avoid division by zero and extreme z values
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in extreme cases
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an exploration bonus proportional to uncertainty
    # This helps escape local minima and explore high-uncertainty regions
    exploration_bonus = 1.0 * sigma_safe
    
    # Cost-aware: divide by cost
    # cost has shape (2000,) and is positive (1 + mean(x_scaled))
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
