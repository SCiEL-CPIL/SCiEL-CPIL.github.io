import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: Scalar, the lowest observed value.
        cost: Array of shape (2000,) with the cost of evaluating each candidate.
        
    Returns:
        Array of shape (2000,) with acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    safe_cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample one function value per candidate
    # We want to minimize, so lower sampled values are better
    y = np.random.normal(loc=mu, scale=sigma)
    
    # Cost-aware scoring:
    # We want to maximize the score. Lower y is better (minimization), lower cost is better.
    # A common cost-aware metric is to maximize the expected improvement per unit cost,
    # or to use a penalty term.
    #
    # Let's use a formulation that balances exploration and cost.
    # Score = -y / cost  (maximize this)
    # This favors points with low predicted value and low cost.
    #
    # However, this can be too aggressive on cost. Let's try a softer penalty.
    # Alternative: Score = -y - lambda * cost, where lambda adapts.
    # Or: Score = -(y - f_best) / (cost + 1)  for points that might improve.
    #
    # Let's stick with a simple but effective cost-aware Thompson sampling:
    # We sample y ~ N(mu, sigma^2). The "value" of evaluating this point is -y.
    # The "cost" is cost. We want to maximize value/cost or value - penalty*cost.
    #
    # A robust approach: maximize -y / (1 + cost) or similar.
    # Since cost is in [1, 2] (1 + mean(x in [0,1])), dividing by cost is reasonable.
    
    scores = -y / safe_cost
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
