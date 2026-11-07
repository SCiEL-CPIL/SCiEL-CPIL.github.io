"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending PI and EI with exponential cost penalty.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Blend PI and EI: weight PI more heavily for sharper preference
    # This creates a stronger preference for points that are very likely to improve
    blend = 0.6 * pi + 0.4 * ei / (np.max(ei) + 1e-9)
    
    # Apply exponential cost penalty to strongly prefer cheap evaluations
    # The cost is in range [1, 2] based on the problem description
    # We use an exponential penalty to create a sharper preference for low-cost points
    cost_penalty = np.exp(-0.5 * (cost - 1.0))
    
    # Final score: blended improvement metric weighted by cost penalty
    scores = blend * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
