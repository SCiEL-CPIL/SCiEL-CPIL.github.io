"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # UCB-style exploration term
    # Use a fixed beta that balances exploitation and exploration
    # Higher beta = more exploration
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # EI focuses on improvement over f_best
    # UCB balances mean and uncertainty
    combined = 0.7 * ei + 0.3 * ucb
    
    # Cost-aware score: divide by cost to prefer cheaper evaluations
    # Use a soft cost penalty: cost^gamma where gamma < 1 to avoid over-penalizing
    gamma = 0.5
    cost_penalty = np.power(cost + 1e-9, gamma)
    
    scores = combined / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
