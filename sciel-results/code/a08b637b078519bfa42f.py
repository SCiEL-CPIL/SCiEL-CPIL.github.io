import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Mean of the GP surrogate, shape (N,)
        sigma: Standard deviation of the GP surrogate, shape (N,)
        f_best: Best observed function value so far (scalar)
        cost: Cost of evaluating each candidate, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher is better.
    """
    # Use a fixed beta for exploration
    beta = 2.0
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate LCB: mu + beta * sigma
    # We want to minimize this value
    lcb = mu + beta * sigma
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Score is negative LCB divided by cost
    # Maximizing this score minimizes LCB/cost
    scores = -lcb / cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
