"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: (2000,) GP mean predictions.
        sigma: (2000,) GP standard deviation predictions.
        f_best: (scalar) Best observed value so far.
        cost: (2000,) Cost of evaluating each candidate.
        
    Returns:
        (2000,) Scores. Higher is better.
    """
    global _start_time
    
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # If we are running out of time, return a safe, deterministic score
        # that favors low mu (exploitation) to ensure we don't waste the last seconds
        # on random exploration.
        return -mu / (cost + 1e-8)

    # UCB for minimization:
    # We want to minimize f(x).
    # The optimistic lower bound is mu - beta * sigma.
    # We want to maximize the potential for improvement, so we score by:
    # Score = (beta * sigma - mu) / cost
    
    # Tuning beta:
    # Beta=2.0 is standard. Beta=3.0 encourages more exploration.
    # Given the success of Thompson Sampling (exploration), we use a higher beta.
    beta = 3.0
    
    # Ensure cost is positive and finite
    c = np.maximum(cost, 1e-8)
    
    # Calculate scores
    scores = (beta * sigma - mu) / c
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
