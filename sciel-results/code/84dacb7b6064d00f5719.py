"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return safely before it.
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to get EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # UCB exploration bonus: balance exploitation and exploration
    # Use a moderate beta that adapts based on uncertainty
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Blend EI and UCB: use EI as primary, add UCB for exploration
    # In early stages (high sigma), UCB helps; later, EI dominates
    # Weight based on relative uncertainty
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-6:
        # Normalize UCB to be comparable to EI scale
        ucb_normalized = ucb / (sigma_mean + 1e-9)
        # Blend: 70% EI, 30% UCB
        scores = 0.7 * ei_cost + 0.3 * (-ucb_normalized) / cost_safe
    else:
        scores = ei_cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time: if we've used more than 325s, return immediately
    if time.time() - t0 > 325:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
