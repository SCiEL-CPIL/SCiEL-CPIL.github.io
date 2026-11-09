"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Lower Confidence Bound (LCB) for minimization.
    
    We use LCB to balance exploration and exploitation.
    The score is defined as -mu + beta * sigma, which we want to maximize.
    This is equivalent to minimizing mu - beta * sigma.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Hyperparameter for exploration
    # A value of 2.0 is a common starting point for balanced exploration
    beta = 2.0
    
    # Calculate LCB score: higher is better
    # LCB for minimization is mu - beta * sigma
    # We want to maximize the negative of this: -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace -inf with a very small number to ensure finite output as per contract
    # The contract says "Return 2000 finite scores"
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
