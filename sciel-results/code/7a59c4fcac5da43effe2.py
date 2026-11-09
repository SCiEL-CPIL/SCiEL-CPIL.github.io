"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound acquisition function for minimization.
    
    For minimization, we want to find points with low predicted mean
    and high uncertainty. The UCB score is:
        score = mu - beta * sigma
    
    Higher scores mean more promising to evaluate next.
    """
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: low mean + high uncertainty is good
    # We negate so that higher = better (as required)
    beta = 3.0
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
