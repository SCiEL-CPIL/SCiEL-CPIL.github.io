"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using log-transformed Expected Improvement.
    
    Structural change: instead of a linear blend of EI and UCB, we use
    log1p(EI) which compresses the dynamic range and changes the relative
    weighting of candidate points. This is a different functional family
    that may better balance exploration and exploitation for this short-horizon
    problem.
    
    mu and sigma are in objective units (lower is better for the objective).
    We want to maximize the acquisition score.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative for log transform
    ei_clipped = np.maximum(ei, 0.0)
    
    # Log-transform to compress dynamic range
    # This makes the acquisition less greedy about large EI values
    # and gives more relative weight to moderately promising points
    score = np.log1p(ei_clipped)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
