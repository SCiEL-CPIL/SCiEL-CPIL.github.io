"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin from 330s

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if _check_deadline():
        # Return a simple heuristic: lowest cost (cheapest to evaluate)
        # Or just zeros? 
        # Returning -cost encourages cheap points if we are late?
        # But we must return finite scores.
        return -cost

    # Lower Confidence Bound (LCB) / Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Minimization: low mu is good. High sigma is good for exploration.
    # Score = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost
    # Handle cost > 0
    scores = ucb / (cost + 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
