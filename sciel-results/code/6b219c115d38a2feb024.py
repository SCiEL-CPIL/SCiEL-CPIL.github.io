"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, leave 50s margin
    
    # Numerical stability: clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Structural Change: Use Upper Confidence Bound (UCB) for minimization.
    # UCB balances exploitation (minimizing mu) and exploration (maximizing sigma).
    # Score to maximize: -mu + beta * sigma
    # For Ackley 2D, a beta of 1.5 provides a good balance.
    beta = 1.5
    scores = -mu + beta * sigma_safe
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
