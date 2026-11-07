# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-penalized Expected Improvement.
    
    Parameters
    ----------
    mu : array-like, shape (n,)
        GP mean predictions.
    sigma : array-like, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost penalty: subtract lambda * cost
    # We want to prefer cheaper points, so we subtract a fraction of the cost
    # The cost is in [1, 2] since it's 1 + mean(x) where x in [0,1]^3
    # A small lambda like 0.1 should be sufficient
    lambda_cost = 0.05
    scores = ei - lambda_cost * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
