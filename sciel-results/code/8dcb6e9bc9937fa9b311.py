"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        # Return a safe fallback: prefer low cost and low mu
        # This should rarely trigger if the loop is fast enough
        return -mu / (cost + 1e-9)

    # Cost-aware UCB for minimization
    # Score = (Exploration Term - Exploitation Term) / Cost
    # We want high scores.
    # Low mu is good. High sigma is good (uncertainty).
    # Cost is bad (penalize).
    
    # Dynamic kappa: Start with high exploration
    # A fixed kappa of 2.0 to 5.0 is common for UCB in BO.
    # Given 8D, we need more exploration.
    kappa = 3.0
    
    # Calculate UCB score
    # ucb = -mu + kappa * sigma
    # To handle numerical stability and ensure finite scores
    ucb = -mu + kappa * sigma
    
    # Normalize by cost to favor cheaper points
    # cost is in [1, 2]
    scores = ucb / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
