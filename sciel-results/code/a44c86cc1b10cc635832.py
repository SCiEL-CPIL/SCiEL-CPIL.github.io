"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a robust expected improvement formulation with adaptive exploration.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Standard expected improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in the acquisition landscape
    exploration = 0.1 * sigma
    
    # Combine EI and exploration, weighted by inverse cost
    score = (ei + exploration) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
