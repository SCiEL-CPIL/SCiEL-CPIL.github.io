"""Rewritten cost-aware acquisition: Cost-Effective Lower Confidence Bound (CE-LCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # mu, sigma, cost: shape (2000,). Return 2000 finite scores (higher = evaluate next).
    # Ensure sigma is non-negative for stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Lower Confidence Bound (LCB) for minimization:
    # We want to minimize f(x). The LCB is mu - kappa * sigma.
    # We want to select points with LOW LCB.
    # To convert to a "higher is better" score, we negate the LCB.
    # score_unscaled = -(mu - kappa * sigma) = kappa * sigma - mu
    
    # Exploration parameter kappa
    # A value of 1.0 is a standard starting point. 
    # In 2D with Matern-5/2, slight over-exploration can help escape local minima.
    kappa = 1.5
    
    # Calculate the cost-effective LCB score
    # We divide by cost to prefer cheaper evaluations.
    score = (kappa * sigma - mu) / cost
    
    # Safety: keep everything finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    
    return score
# EVOLVE-BLOCK-END
