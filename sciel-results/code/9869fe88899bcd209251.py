import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound).
    """
    t_start = time.time()
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Dynamically tune kappa based on remaining budget
    # Remaining budget = 30 - cost (total cost is sum of all evaluations so far)
    total_cost = np.sum(cost)
    remaining_budget = 30.0 - total_cost
    
    # When remaining budget is high, explore more (higher kappa)
    # When remaining budget is low, exploit more (lower kappa)
    # Use a smooth transition
    if remaining_budget > 0:
        # kappa ranges from ~1.5 (high exploration) to ~0.5 (low exploration)
        # As remaining_budget goes from 30 to 0
        kappa = 0.5 + 1.0 * (remaining_budget / 30.0)
    else:
        kappa = 0.5
    
    # UCB score: mu + kappa * sigma
    ucb = mu + kappa * sigma_safe
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-10)
    scores = ucb / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
        
    return scores
