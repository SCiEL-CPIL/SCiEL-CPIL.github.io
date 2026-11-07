"""Rewritten program with UCB-based cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean predictions from GP (shape: 2000,)
        sigma: Standard deviation predictions from GP (shape: 2000,)
        f_best: Best observed value so far (scalar)
        cost: Cost of evaluating each candidate (shape: 2000,)
        
    Returns:
        scores: Acquisition scores (shape: 2000,), higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost (though cost >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB parameter: balances exploration (sigma) and exploitation (mu)
    # Higher kappa means more exploration.
    kappa = 2.0
    
    # For minimization, we want to maximize the potential for low values.
    # The lower bound on the function value is mu - kappa * sigma.
    # We want to maximize the "improvement potential", which is related to - (mu - kappa * sigma).
    # Score = (kappa * sigma - mu) / cost
    
    scores = (kappa * sigma - mu) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
