"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Draws a single random sample from the posterior at each candidate point:
        theta = mu + sigma * Z,  where Z ~ N(0, 1)
    Returns -theta as the score (higher is better for selection).
    
    This stochastically balances exploration and exploitation:
    - Where sigma is large (uncertain), the sample can swing widely, encouraging exploration.
    - Where mu is low (promising), the sample tends to be low, encouraging exploitation.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Generate standard normal noise for all 2000 candidates
    z = np.random.standard_normal(size=mu.shape)
    
    # Posterior sample: theta = mu + sigma * z
    theta = mu + sigma * z
    
    # For minimization, we want the lowest theta; score is -theta (higher = better)
    scores = -theta
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
