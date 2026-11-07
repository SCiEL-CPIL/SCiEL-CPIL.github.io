"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Bayesian optimization on Ackley 2D.
    
    Uses Thompson Sampling with a stochastic offset to balance exploration and exploitation,
    normalized by cost to account for the cost structure.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: sample a scalar offset from Gaussian centered at f_best
    # with std scaling with maximum uncertainty in the candidate set
    max_sigma = np.max(sigma)
    if not np.isfinite(max_sigma) or max_sigma < 1e-6:
        max_sigma = 1e-6
    
    # Scale the std of the offset to balance exploration/exploitation
    offset_std = 0.5 * max_sigma
    offset = np.random.normal(f_best, offset_std)
    
    # Score: -(mu - offset) / cost = (offset - mu) / cost
    # This stochastically shifts the objective, favoring low mu (exploitation)
    # but the random offset allows exploration when offset > f_best
    score = (offset - mu) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
