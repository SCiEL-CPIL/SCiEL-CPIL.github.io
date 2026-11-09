import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Args:
        mu: Mean of the GP, shape (2000,)
        sigma: Standard deviation of the GP, shape (2000,)
        f_best: Best observed value so far
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: 2000 finite scores, higher is better.
    """
    # UCB for minimization: Maximize (-mu + beta * sigma)
    # Penalize high cost: Subtract lambda * cost
    
    # Parameters
    # beta: Exploration constant. 2.5 is a reasonable default for 6D.
    # lambda_cost: Cost penalty. 1.0 balances cost (1-2) with uncertainty (sigma).
    beta = 2.5
    lambda_cost = 1.0
    
    # Calculate scores
    scores = -mu + beta * sigma - lambda_cost * cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
