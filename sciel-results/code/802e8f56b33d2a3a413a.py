import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    Balances exploitation (low mu) and exploration (high sigma)
    while penalizing evaluation cost.
    
    Args:
        mu: Mean predictions (2000,)
        sigma: Standard deviation predictions (2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        scores: Acquisition scores (2000,), higher is better.
    """
    # UCB for minimization: we want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    # Cost-aware: divide by cost
    
    # Beta controls the exploration-exploitation trade-off
    # With a small budget, we need sufficient exploration
    beta = 2.5
    
    # Calculate UCB score
    # Using safe sigma to avoid issues, though UCB is less sensitive to sigma=0 than EI
    ucb_raw = beta * sigma - mu
    
    # Cost-aware score
    # We divide by cost to penalize expensive points
    # cost is in [1, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = ucb_raw / cost
        
    # Replace NaN and Inf with finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to reasonable range to prevent overflow issues in downstream logic
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
