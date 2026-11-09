"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: array of shape (2000,) GP mean predictions.
        sigma: array of shape (2000,) GP standard deviation predictions.
        f_best: float, best observed function value so far.
        cost: array of shape (2000,) costs for each candidate.
        
    Returns:
        array of shape (2000,) scores, higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in cost, though cost is >= 1
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB parameter (exploration coefficient)
    # A value of 1.5 balances exploitation and exploration well for this scale
    beta = 1.5
    
    # Lower Confidence Bound: mu - beta * sigma
    # We want to minimize this, so we maximize its negative.
    # Score = (beta * sigma - mu) / cost
    scores = (beta * sigma - mu) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
