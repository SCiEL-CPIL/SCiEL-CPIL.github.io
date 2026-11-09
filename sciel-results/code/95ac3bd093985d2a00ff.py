"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a Probability of Improvement (PoI) confidence factor.
    
    Score = (EI / cost) * (1 + PoI)
    
    This amplifies scores for points that are both likely to improve
    (high PoI) and have high expected improvement, while suppressing
    scores for high-variance points with low probability of improvement.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative cost
    cost = np.maximum(cost, 1e-9)
    
    # Standard deviation floor to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Combine: cost-normalized EI amplified by PoI confidence factor
    # The (1 + PoI) factor gives extra weight to points with high probability of improvement
    score = (ei / cost) * (1.0 + poi)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
