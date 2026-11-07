"""
Bayesian Optimization Acquisition Function
Strategy: Upper Confidence Bound (UCB) for minimization.
This is a structural change from the seed Expected Improvement (EI).
UCB balances exploration (high sigma) and exploitation (low mu) with a fixed coefficient.
"""
import numpy as np
import time

# Global time tracking for deadline guard
_START_TIME = time.time()
# The evaluator kills the program after 330s.
# We leave a 30s margin for safety.
_TIME_LIMIT = 300.0

def _check_deadline():
    """Returns True if we are still within the time limit."""
    return (time.time() - _START_TIME) < _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Computes the UCB score for each candidate.
    Higher score means more promising to evaluate next.
    
    Args:
        mu: Posterior mean at candidates, shape (2000,)
        sigma: Posterior std dev at candidates, shape (2000,)
        f_best: Current best observed value (lowest).
        
    Returns:
        scores: Array of shape (2000,) with scores to maximize.
    """
    # Deadline guard: 
    # If we are close to the limit, we could return a conservative score 
    # (e.g., just -mu) to ensure we finish quickly, but the op is O(N) and fast.
    if not _check_deadline():
        # Fallback: Exploit only (lowest mean)
        return -mu

    # UCB for minimization:
    # We want to find the point with the lowest potential value.
    # Optimistic lower bound: mu - beta * sigma
    # To maximize the score, we return the negative of the lower bound:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    return scores
