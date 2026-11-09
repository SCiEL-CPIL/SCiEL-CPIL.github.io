"""Rewritten program using Cost-Aware UCB (Upper Confidence Bound) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety guard
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Upper Confidence Bound (UCB).
    
    Args:
        mu: Array of GP means, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: Scalar, best observed function value so far
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        Array of scores, shape (N,)
    """
    # Time guard: if we are close to the limit, return a safe, simple score
    # to avoid timeout. In practice, this function is very fast, so this is 
    # mostly a safety net.
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to a simple exploration score
        return np.ones_like(mu) * (f_best - np.mean(mu) + np.mean(sigma)) / np.mean(cost)

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)

    # UCB parameter (beta). 
    # The knowledge base suggests linear regret (beta=1) worked well (1.0 -> 1.713).
    # We use beta=3.0 to encourage more exploration of high-uncertainty regions,
    # which is structurally different from the exploitative EI.
    beta = 3.0
    
    # Calculate UCB: (f_best - mu) is expected improvement in mean.
    # + beta * sigma adds exploration bonus.
    # Dividing by cost makes it cost-aware (higher cost penalizes score).
    ucb_term = (f_best - mu) + beta * sigma
    score = ucb_term / cost
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
