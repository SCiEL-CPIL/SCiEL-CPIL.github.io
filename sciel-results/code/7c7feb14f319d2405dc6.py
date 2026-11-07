"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # UCB with linear cost penalty
    # beta controls exploration vs exploitation
    beta = 2.0
    # Cost normalization: cost is in [1, 2], so we can use it directly
    # UCB = mu + beta * sigma - gamma * cost
    # For minimization, we want low mu, high sigma (uncertainty), low cost
    # Since higher score = evaluate next, we want to maximize the acquisition
    # So: acquisition = -mu + beta * sigma - gamma * cost
    # But we need to be careful with scaling
    
    # Let's use a simpler approach: 
    # Score = -mu + beta * sigma - cost_penalty
    # where cost_penalty accounts for the cost
    
    # From the knowledge: UCB (beta=2) with linear cost penalty improved from 0.9976 to 1.034
    # So: acquisition = -mu + beta * sigma - gamma * cost
    
    # Let's try gamma = 1.0 for linear cost penalty
    gamma = 1.0
    
    # Handle zero/near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to minimize, so lower mu is better
    # But acquisition should be higher for more promising points
    # So: acquisition = -(mu - beta * sigma) - gamma * cost
    #    = -mu + beta * sigma - gamma * cost
    
    acquisition_scores = -mu + beta * sigma_safe - gamma * cost
    
    # Ensure finite values
    acquisition_scores = np.where(np.isfinite(acquisition_scores), acquisition_scores, -np.inf)
    
    return acquisition_scores
# EVOLVE-BLOCK-END
