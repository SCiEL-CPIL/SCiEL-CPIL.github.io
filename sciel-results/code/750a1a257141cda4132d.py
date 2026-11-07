"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin
    
    # Calculate Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x). LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the "promise" of being low, so we maximize -LCB(x)
    # -LCB(x) = -mu(x) + beta * sigma(x)
    # Cost-aware score: (-mu(x) + beta * sigma(x)) / cost(x)
    
    beta = 1.0  # Exploration-exploitation trade-off parameter
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate LCB-based score
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    scores = lcb_score / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Safety margin check
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
