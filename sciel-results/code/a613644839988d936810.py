"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Use a cost-aware Upper Confidence Bound (UCB) variant
    # This is often more robust than EI when the GP model is uncertain.
    # Score = (Expected Improvement proxy) / Cost
    # Proxy: (f_best - mu) + beta * sigma
    # We want to maximize this score.
    
    beta = 1.0
    
    # Calculate the optimistic improvement
    # If mu is much lower than f_best, (f_best - mu) is large positive.
    # If sigma is large, beta * sigma adds to the score, encouraging exploration.
    optimistic_improvement = (f_best - mu) + beta * sigma
    
    # Cost-aware normalization
    # Divide by cost to penalize expensive points.
    scores = optimistic_improvement / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
