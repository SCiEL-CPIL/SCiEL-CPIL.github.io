"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if time is running out, return a safe score (cheapest candidate)
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(cost, 1e-6)
        return 1.0 / cost_safe

    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Higher score means more promising (lower expected value or high uncertainty)
    beta = 1.0
    score = -mu + beta * sigma
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-6)
    score = score / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
