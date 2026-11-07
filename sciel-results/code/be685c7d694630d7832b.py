"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_deadline = None


def acquisition(mu, sigma, f_best):
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 320.0  # safety margin before 330s limit
    
    # Compute standard Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus (soft UCB) to help escape local optima
    # This stays within the EI family by using EI as the base score
    beta = 0.1
    exploration = beta * sigma_safe
    
    scores = ei + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
