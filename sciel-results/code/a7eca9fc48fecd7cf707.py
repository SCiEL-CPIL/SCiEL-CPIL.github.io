"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean
    sigma : array of shape (2000,) - GP std deviation
    f_best : float - best observed value
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=np.float64)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Structural Change: Use Cost-Aware UCB
    # Score = ( -mu + beta * sigma ) / cost
    # This directly optimizes for low mean (exploitation) and high uncertainty (exploration)
    # while penalizing high cost.
    
    # Determine beta. 
    # A fixed beta is more robust than adaptive ones in short runs.
    # The problem is on [-5.12, 5.12]^2. Rastrigin has a maximum value around 100+.
    # However, mu and sigma are in "objective units".
    # Standard practice for UCB in BO is beta ~ 1-5. 
    # Let's use a slightly conservative exploration factor.
    beta = 2.0
    
    # Calculate UCB score (higher is better)
    ucb_raw = -mu + beta * sigma
    
    # Cost-aware score
    scores = ucb_raw / cost_safe
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf or similar, return zeros to avoid issues
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu, dtype=np.float64)
        
    return scores
# EVOLVE-BLOCK-END
