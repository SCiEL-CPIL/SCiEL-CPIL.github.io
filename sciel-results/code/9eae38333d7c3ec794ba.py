import numpy as np
import time

# Global variables to track time if needed, though acquisition is fast
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: (2000,) array of GP means in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we use (beta * sigma - mu)
    # This is equivalent to minimizing (mu - beta * sigma)
    
    # Beta controls exploration. 2.0 is a common aggressive value.
    beta = 2.0
    
    # Calculate LCB score
    # We use the "optimistic" value: mu - beta * sigma
    # Higher score means lower expected value with high confidence
    lcb_values = mu - beta * sigma
    
    # To make it a "score" where higher is better for selection:
    # We want to pick the point with the lowest LCB value.
    # So score = -lcb_values / cost
    # Or equivalently: score = (beta * sigma - mu) / cost
    
    scores = (beta * sigma - mu) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
