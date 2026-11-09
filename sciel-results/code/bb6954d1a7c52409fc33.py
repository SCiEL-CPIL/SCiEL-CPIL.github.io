"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_start_time = time.time()
# Set a deadline slightly before the hard kill to allow for safe return
# 330s is the kill time. We use 325s to be safe.
_DEADLINE = _start_time + 325.0 

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    # Ensure inputs are numpy arrays for consistent operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: If we are close to the limit, return a fast, conservative score
    # to ensure we finish safely. -mu/cost prioritizes exploitation of the current
    # best region while respecting cost.
    if time.time() > _DEADLINE - 2.0:
        scores = -mu / np.where(cost > 1e-9, cost, 1.0)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f(x).
    # The "upper confidence bound" on the *minimum* value is mu - beta * sigma.
    # We want to maximize the potential to find a lower value.
    # So the score should be high where mu - beta * sigma is low.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages exploration of high uncertainty regions.
    # For 8D problems, a moderate beta is often effective.
    # We use a fixed beta of 2.5 which is a common heuristic for BO.
    beta = 2.5
    
    # Calculate UCB score
    # Higher score means "more promising" (lower expected minimum)
    ucb_score = -mu + beta * sigma
    
    # Cost-awareness: Divide by cost to prioritize cheaper evaluations.
    # Maximizing ucb_score / cost.
    scores = ucb_score / cost_safe
    
    # Ensure all scores are finite. If not, replace with 0.0 (neutral/invalid).
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
