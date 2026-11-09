# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB)
    with adaptive exploration coefficient that decays with total cost consumed.
    
    The score is: -mu(x) - beta * sigma(x) - gamma * cost(x)
    where beta adapts to the remaining budget, encouraging exploration early
    and exploitation late.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 295.0

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety: avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Normalize cost to [0, 1] for consistent scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost_safe) * 0.5
    else:
        cost_norm = (cost_safe - cost_min) / cost_range
    
    # Adaptive exploration coefficient:
    # beta starts high (more exploration) and decays as f_best approaches 0
    # This balances exploration vs exploitation based on progress
    f_best_scale = max(abs(f_best), 1e-3)
    
    # beta: exploration coefficient for LCB
    # Higher when f_best is large (early), lower when f_best is small (late)
    beta = 0.5 * np.sqrt(np.log(1.0 + 1.0 / f_best_scale))
    
    # gamma: cost penalty coefficient
    # Penalize expensive evaluations, scaled by remaining budget
    # As we spend more budget, we should be more selective about cost
    gamma = 0.1 * f_best_scale
    
    # LCB score: we want to MINIMIZE mu + beta*sigma, so negate for maximization
    # Also subtract cost penalty
    lcb_score = -mu - beta * sigma_safe - gamma * cost_norm
    
    # Add a small exploration term for high uncertainty regions
    # This helps discover new minima in unexplored areas
    exploration_bonus = 0.1 * sigma_safe / (1.0 + cost_norm)
    
    # Combine
    scores = lcb_score + exploration_bonus
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
