import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # We want to find the minimum.
    # LCB = mu - beta * sigma
    # We want to minimize LCB, which is equivalent to maximizing -LCB = beta * sigma - mu.
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Exploration parameter
    # A value of 2.0 is a standard choice for Bayesian Optimization UCB/LCB.
    beta = 2.0
    
    # Calculate score
    # Higher score indicates a better candidate to evaluate next
    scores = beta * sigma - mu
    
    # Cost-aware normalization
    # Cost is in [1, 2] (1 + mean(x) where x in [0,1])
    # Dividing by cost prioritizes cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = scores / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
