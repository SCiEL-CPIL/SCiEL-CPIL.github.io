import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition function.
    
    For minimization, we want low mu and high uncertainty (sigma).
    LCB = mu - beta * sigma
    Cost-aware score: we want to maximize the "improvement potential" per unit cost.
    
    Using: score = (f_best - mu + beta * sigma) / cost
    This combines exploitation (f_best - mu) with exploration (beta * sigma)
    and normalizes by cost.
    """
    # Adaptive beta: start with more exploration, reduce as f_best improves
    # f_best starts high (~50-100 for Rosenbrock) and decreases toward 0
    # Normalize f_best to get a progress measure
    # Use a simple heuristic: beta decreases as f_best decreases
    
    # For Rosenbrock, initial f_best is typically O(10-100)
    # As we converge, f_best -> 0
    # beta = 1.0 / (1 + f_best) would give beta ~ 1 initially, ~ 1 when f_best is small
    # Actually, let's use a fixed beta that works well
    
    beta = 1.0
    
    # LCB-style score for minimization with cost awareness
    # Higher score = better candidate
    # We want low mu (exploitation) and high sigma (exploration)
    # (f_best - mu) captures potential improvement
    # beta * sigma captures uncertainty bonus
    
    score = (f_best - mu + beta * sigma) / cost
    
    return score
