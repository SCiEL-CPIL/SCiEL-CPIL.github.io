"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    # GP-LCB acquisition with cost-awareness and time guard
    t0 = time.time()
    deadline = 295.0  # safety margin before 300s limit
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB: mu - beta * sigma, normalized by cost
    # beta controls exploration-exploitation tradeoff
    # For cost-aware, we want to balance exploration with cost efficiency
    beta = 2.0
    
    lcb = mu - beta * sigma_safe
    
    # Cost-aware scaling: prefer cheaper evaluations
    # cost is in [1, 2] (1 + mean of scaled coords)
    # Normalize cost to [0, 1] range for better conditioning
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range
    
    # Acquisition: LCB per unit cost, with slight exploration bonus
    # Higher score = more promising
    scores = -lcb / (1.0 + cost_norm * 0.5)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - t0
    if elapsed > deadline:
        # Fallback: return simple heuristic if we're running out of time
        return -mu / cost
    
    return scores
# EVOLVE-BLOCK-END
