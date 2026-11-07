"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition: select the candidate that minimizes
    the lower confidence bound (mu - k*sigma).
    
    We return scores where higher is better, so we return the negative
    of the lower confidence bound.
    
    To make this more effective, we use a time-adaptive k that increases
    as we get closer to the optimum (when f_best is close to the minimum).
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a moderate exploration parameter
    # The lower confidence bound is mu - k*sigma
    # We want to minimize this, so we return -(mu - k*sigma) = -mu + k*sigma
    # Higher score means we prefer lower (mu - k*sigma)
    
    # Adaptive k: use a value that balances exploration and exploitation
    # For 1D problems, k=2.0 is a reasonable choice
    k = 2.0
    
    # Lower confidence bound
    lcb = mu - k * sigma
    
    # Return negative LCB so that higher scores correspond to lower LCB values
    scores = -lcb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
