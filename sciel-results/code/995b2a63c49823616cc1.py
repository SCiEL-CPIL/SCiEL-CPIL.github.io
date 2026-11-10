import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Parameters:
    mu: (2000,) GP mean
    sigma: (2000,) GP standard deviation
    f_best: float, best observed value
    cost: (2000,) cost of evaluating each candidate
    
    Returns:
    scores: (2000,) acquisition scores, higher is better
    """
    eps = 1e-9
    
    # UCB parameter: balance exploration vs exploitation
    # Higher beta = more exploration of uncertain regions
    beta = 1.0
    
    # UCB score: mu + beta * sigma (for minimization, we want low mu, high sigma)
    # Normalize by cost for cost-awareness
    ucb = mu + beta * sigma
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = -ucb / (cost + eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
