"""Acquisition function for Bayesian optimization of Goldstein-Price 2D (minimization)."""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition for minimization.
    Returns scores where higher means "evaluate next".
    score = P(f(x) < f_best)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-9
    safe_sigma = np.where(sigma > eps, sigma, eps)
    
    # Calculate the Z-score for the probability of improvement
    # We want P(f < f_best) where f ~ N(mu, sigma^2)
    # P(f < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / safe_sigma
    
    # Compute the CDF of the standard normal distribution
    score = norm.cdf(z)
    
    # For very small sigma, the probability is effectively 1 if mu < f_best, else 0
    small_sigma_mask = sigma < eps
    if np.any(small_sigma_mask):
        score[small_sigma_mask] = np.where(mu[small_sigma_mask] < f_best, 1.0, 0.0)
        
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
