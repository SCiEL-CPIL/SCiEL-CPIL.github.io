"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 320  # Seconds, leaving 10s buffer for the 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: -mu + beta * sigma
    # beta=1.0 was found effective for Griewank's multimodal landscape
    beta = 1.0
    ucb_scores = -mu + beta * sigma_safe
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend UCB and EI with equal weight
    # UCB provides linear exploration for multimodal navigation
    # EI provides exploitation of expected gain
    blended_scores = 0.5 * ucb_scores + 0.5 * ei_scores
    
    # Cost-aware score: Higher score = better candidate
    # We want to maximize blended acquisition per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = blended_scores / cost_safe
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
