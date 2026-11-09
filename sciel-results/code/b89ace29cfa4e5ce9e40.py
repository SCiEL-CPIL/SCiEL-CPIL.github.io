import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 290:
        z = (f_best - mu) / (sigma + 1e-9)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0)
        return EI / (cost + 1e-9)

    # UCB-like hybrid: f_best - mu + beta * sigma
    # This encourages exploration by rewarding high uncertainty (sigma)
    # even when the mean prediction (mu) is worse than f_best.
    # Beta controls the exploration-exploitation trade-off.
    beta = 1.5
    scores = (f_best - mu) + beta * sigma
    
    # Cost-aware scaling: divide by cost to prefer cheaper evaluations
    # Add small epsilon to avoid division by zero
    scores = scores / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
