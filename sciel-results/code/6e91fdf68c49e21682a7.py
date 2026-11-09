import numpy as np

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
    # Convert to numpy arrays to ensure operations are vectorized
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle potential edge cases where sigma might be 0 or negative due to numerical errors
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample a realization of the function value for each candidate
    # y ~ N(mu, sigma^2)
    y = np.random.normal(loc=mu, scale=sigma)
    
    # Cost-aware scoring:
    # We want to minimize y, but we also want to minimize cost to save budget.
    # The score -y / cost favors candidates with low predicted value (low y) 
    # and low evaluation cost.
    # This transforms the problem into maximizing "value per unit cost".
    
    # Ensure cost is not zero (though it should be >= 1)
    safe_cost = np.maximum(cost, 1e-9)
    
    scores = -y / safe_cost
    
    # Ensure all scores are finite
    scores[~np.isfinite(scores)] = -np.inf
    
    return scores
