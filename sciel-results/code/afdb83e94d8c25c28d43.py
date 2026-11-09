"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: PoI/cost + uncertainty bonus.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = more desirable to evaluate).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # Under GP: f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Cost-adjusted PoI
    cost_poi = poi / cost
    
    # Uncertainty bonus: reward exploring where sigma is high,
    # but down-weight by cost
    # Use a normalized uncertainty term
    sigma_max = np.max(sigma)
    if sigma_max > 0:
        uncertainty_bonus = (sigma / sigma_max) / cost
    else:
        uncertainty_bonus = np.zeros_like(sigma)
    
    # Combine: primarily cost-adjusted PoI, with a small uncertainty bonus
    # The uncertainty bonus helps explore in early iterations when
    # the GP is uncertain everywhere
    scores = cost_poi + 0.1 * uncertainty_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
