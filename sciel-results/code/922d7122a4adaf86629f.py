"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard: return fallback if we're getting close to timeout
    # The main computation should be fast, but let's be safe
    deadline = start_time + 290  # 290s margin from 300s limit
    
    # Standard EI per unit cost
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_cost = ei / cost
    
    # Also compute UCB-style scores for blending
    # UCB: f_best - mu + kappa * sigma, divided by cost
    # This balances exploitation and exploration differently than EI
    
    # Try a few kappa values and blend with EI
    # Use a weighted combination that favors EI but adds some UCB for exploration
    kappa = 1.0
    ucb = (f_best - mu) + kappa * sigma_safe
    ucb_cost = ucb / cost
    
    # Blend: primarily EI, with a small UCB component to encourage exploration
    # The blend weight can be tuned, but let's start with 0.9 EI + 0.1 UCB
    score = 0.9 * ei_cost + 0.1 * ucb_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return EI cost as fallback
        return ei_cost
    
    return score
# EVOLVE-BLOCK-END
