import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Array of GP means.
        sigma: Array of GP standard deviations.
        f_best: Current best observed value.
        cost: Array of costs for each candidate.
        
    Returns:
        Array of acquisition scores.
    """
    # Ensure cost is positive to avoid division errors
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB parameter: balances exploration (sigma) and exploitation (-mu)
    # A higher beta encourages more exploration.
    # Given the small budget, exploration is crucial.
    beta = 1.5
    
    # Calculate UCB score
    # We want to minimize f, so we maximize -mu + beta * sigma
    ucb_scores = -mu + beta * sigma
    
    # Normalize by cost to prefer cheaper evaluations
    scores = ucb_scores / cost_safe
    
    return scores
