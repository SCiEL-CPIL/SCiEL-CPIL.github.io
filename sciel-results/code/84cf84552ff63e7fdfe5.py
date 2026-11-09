"""Rewritten acquisition for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute Lower Confidence Bound (LCB) acquisition score.
    
    For minimization, the LCB is mu - xi * sigma.
    We want to minimize this, so we return the negative as the score
    (higher score = better).
    """
    # Exploration-exploitation trade-off parameter
    xi = 1.0
    
    # LCB score for maximization interface: -mu + xi * sigma
    score = -mu + xi * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
