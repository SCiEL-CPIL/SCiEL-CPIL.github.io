"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Lower Confidence Bound (LCB) for minimization.
    # We maximize the score.
    # Score = (f_best - (mu - beta * sigma)) / cost
    # This is equivalent to maximizing (f_best - mu + beta * sigma) / cost
    
    # Beta controls the exploration-exploitation trade-off.
    # Higher beta encourages exploration (higher sigma).
    # For multimodal Shekel 4d, a moderate beta is needed to escape local optima.
    beta = 2.5
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate LCB value: mu - beta * sigma
    # We want to minimize this, so we maximize -LCB = -mu + beta * sigma
    # Relative to f_best: f_best - mu + beta * sigma
    
    numerator = f_best - mu + beta * sigma
    denominator = np.maximum(cost, 1e-9)
    
    scores = numerator / denominator
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
