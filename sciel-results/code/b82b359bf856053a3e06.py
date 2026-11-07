"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 290.0  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - START_TIME > TIME_LIMIT:
        # Return a safe fallback: uniform random scores or just EI
        # Since we must return 2000 finite scores, we return a simple score
        # to avoid invalidating the result.
        # Using a random score might be bad. Using EI is safer.
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost

    # Cost-aware UCB
    # Score = (f_best - mu + beta * sigma) / cost
    # We clip the numerator to be non-negative to avoid selecting points
    # that are certain to be worse than f_best.
    beta = 2.0
    improvement = f_best - mu + beta * sigma
    improvement = np.maximum(improvement, 0.0)
    
    # Avoid division by zero or negative cost (cost is always >= 1)
    scores = improvement / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
