"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use a combination of Expected Improvement and a small exploration term
    # to balance exploitation and exploration
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    # Compute Expected Improvement
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small UCB-like exploration term to encourage exploration
    # beta controls exploration-exploitation tradeoff
    beta = 0.1
    ucb = -mu + beta * sigma
    
    # Combine: primarily EI with a small UCB component
    # This helps escape local optima while still exploiting promising regions
    scores = ei + 0.01 * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're close to deadline - if so, return immediately
    if time.time() > deadline - 5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
