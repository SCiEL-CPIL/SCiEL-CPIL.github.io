"""Cost-aware UCB acquisition for Hartmann 6D minimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: -mu + beta * sigma
    # Linear cost penalty as per directive
    # Higher beta encourages exploration to counteract linear cost penalty
    beta = 2.5
    
    # Base UCB score for minimization
    base_score = -mu + beta * sigma
    
    # Apply linear cost penalty
    scores = base_score / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
