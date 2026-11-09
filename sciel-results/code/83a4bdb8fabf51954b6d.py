"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time to track deadline
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s safety margin within 330s limit

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB (Upper Confidence Bound) acquisition function.
    
    Strategy:
    Instead of Expected Improvement (EI) which is exploitation-heavy, we use UCB
    which balances exploitation (mu) and exploration (sigma) with a tunable beta.
    We divide by cost to prioritize cheaper evaluations.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: scalar, best observed value (lowest)
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety check for time
    if _time_remaining() < 1.0:
        # If running out of time, return a simple heuristic quickly
        # Just return mu/cost to avoid complex computations
        return mu / (cost + 1e-9)

    # Avoid division by zero
    safe_cost = np.maximum(cost, 1e-6)
    
    # UCB Parameter:
    # A fixed beta is used. For 2D problems with Matern-5/2, 
    # beta around 2.0-5.0 often works well. 
    # We can make it slightly adaptive if we had step count, but we don't.
    # However, we can estimate progress from f_best if it's improving.
    # For now, a fixed beta of 3.0 encourages exploration.
    beta = 3.0
    
    # Compute UCB
    ucb = mu + beta * sigma
    
    # Cost-aware normalization
    scores = ucb / safe_cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
