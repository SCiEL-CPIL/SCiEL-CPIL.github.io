# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB: balance exploitation and exploration, penalize by cost
    # f_best is the best (lowest) observed value; we want to minimize
    # So improvement = f_best - mu (positive when mu < f_best)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB-style: we want high values where mu is low and sigma is high
    # (f_best - mu) is the expected improvement in mean
    # Add exploration bonus: beta * sigma
    # Divide by cost to account for evaluation cost
    
    # Adaptive beta: use a fixed value that balances exploration/exploitation
    beta = 1.5
    
    # Score: higher is better
    # (f_best - mu) + beta * sigma, all divided by cost
    score = ((f_best - mu) + beta * sigma) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
