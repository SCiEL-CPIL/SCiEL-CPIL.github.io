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
    
    # Thompson Sampling: sample from GP posterior for exploration
    # Sample a single value from the posterior
    eps = np.random.randn()
    ts_sample = mu + eps * sigma_safe
    
    # Cost-aware: prefer lower cost regions, but allow exploration
    # Use a softer cost penalty that still favors cheap regions
    # cost is in [1, 2] since it's 1 + mean(x_scaled)
    # We want to strongly prefer low-cost but not completely ignore high-cost
    cost_penalty = np.exp(-0.5 * (cost - 1.0))
    
    # Combine Thompson sample with cost preference
    # Lower ts_sample is better (minimization), so we negate
    # Add exploration bonus for uncertain regions
    exploration_bonus = sigma_safe * 0.1
    
    scores = -(ts_sample) * cost_penalty + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
