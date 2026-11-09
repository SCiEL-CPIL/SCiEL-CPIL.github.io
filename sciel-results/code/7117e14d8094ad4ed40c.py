"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB for exploration and POI for exploitation.
    Uses rank-based normalization for robustness against outliers and scale changes.
    """
    t_start = time.time()
    
    # Safety margin for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # Increased beta to 2.5 for stronger exploration in multi-modal Branin
    beta = 2.5
    ucb = -mu + beta * sigma
    
    # Normalize UCB using rank-based method for robustness
    # This ensures that the scores are in [0, 1] and relative ordering is preserved
    # without being skewed by extreme outliers in mu or sigma.
    n = len(ucb)
    
    # Calculate ranks (1 to n)
    ucb_ranks = np.argsort(np.argsort(ucb)) / n
    poi_ranks = np.argsort(np.argsort(poi)) / n
    
    # Combine: 70% UCB, 30% POI
    # Higher weight on UCB to encourage exploration of the 7 local minima
    score = 0.7 * ucb_ranks + 0.3 * poi_ranks
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
