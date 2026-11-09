# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # Score = -mu + beta * sigma
    # Higher score indicates a better candidate to evaluate.
    # -mu favors points with low predicted mean (exploitation).
    # beta * sigma favors points with high uncertainty (exploration).
    
    # Clip sigma to avoid numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Exploration parameter beta
    # A higher beta encourages more exploration.
    # For multimodal functions, a moderate beta is often effective.
    beta = 2.0
    
    # Calculate UCB scores
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
