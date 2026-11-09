"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based cost-discounted Lower Confidence Bound.
    
    Instead of linear EI, we use a quantile of the posterior predictive
    (5th percentile) to represent the expected low value, and apply a
    non-linear sigmoidal cost penalty to aggressively discount high-cost
    candidates.
    """
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(len(mu))
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Quantile-based value: 5th percentile of N(mu, sigma^2)
    # This is mu - 1.645 * sigma (5th percentile of standard normal is -1.645)
    quantile_value = mu - 1.645 * sigma_safe
    
    # Sigmoidal cost discount: penalize high-cost candidates more aggressively
    # Cost ranges from 1.0 to 2.0 (since mean(x) in [0,1] adds 0 to 1)
    # We want to discount costs above ~1.3 more strongly
    # Sigmoid centered at 1.3 with steepness 10
    cost_center = 1.3
    cost_steepness = 10.0
    cost_discount = 1.0 / (1.0 + np.exp(cost_steepness * (cost - cost_center)))
    
    # The acquisition score: lower quantile value is better, multiplied by cost discount
    # We want higher scores for candidates that are likely to be low AND have low cost
    # So we use negative quantile value (since lower is better) weighted by cost discount
    score = -quantile_value * cost_discount
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far (the score we just computed)
        # Ensure finite output
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
