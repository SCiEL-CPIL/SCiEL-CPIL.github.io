"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_start_time = time.time()
# Set a deadline slightly before the hard kill to allow for safe return
_DEADLINE = _start_time + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware max-variance UCB acquisition function.
    
    Uses a UCB score that explicitly trades off exploitation (low predicted mean)
    and exploration (high predicted variance), with cost normalization to prefer
    cheaper evaluations when scores are comparable.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        cost_safe = np.where(cost > 1e-9, cost, 1.0)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Max-variance UCB: balance exploitation and exploration
    # Higher sigma encourages exploration of uncertain regions
    # The beta parameter controls the exploration-exploitation tradeoff
    beta = 1.0
    
    # UCB score for minimization: we want low mu and high sigma
    # score = -mu + beta * sigma
    ucb_raw = -mu + beta * sigma_safe

    # Cost-aware normalization: divide by cost to prefer cheaper points
    # This ensures that a point with slightly lower UCB but much lower cost
    # can still be preferred
    scores = ucb_raw / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
