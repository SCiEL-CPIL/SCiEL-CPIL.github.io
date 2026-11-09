import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed value so far (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition values, shape (2000,), higher is better.
    """
    # We want to minimize the function.
    # LCB: mu(x) - beta * sigma(x)
    # We want to maximize the potential for finding a low value, so we maximize -LCB.
    # -LCB = beta * sigma(x) - mu(x)
    
    # Beta controls the exploration-exploitation tradeoff.
    # A value of 2.0 is a common robust choice for small budgets.
    beta = 2.0
    
    # Calculate the cost-aware LCB score
    # Dividing by cost ensures we prefer cheaper points with similar potential.
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
