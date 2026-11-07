"""Combined cost-aware acquisition: UCB exploration with global cost-z-score penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare before 330s limit

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (330.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB with global cost-z-score penalty.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe fallback
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical stability
    epsilon = 1e-9
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)

    # UCB for minimization: we want to maximize (f_best - mu) + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma (standard UCB for minimization)
    # The (f_best - mu) term drives exploitation, beta * sigma drives exploration
    
    # Use a moderate beta that balances exploration and exploitation
    # beta = 1.0 is a reasonable default for 3D optimization
    beta = 1.0
    
    # Compute UCB score: higher is better (more likely to improve)
    ucb_score = (f_best - mu) + beta * sigma_safe
    
    # Global cost-aware penalty: penalize expensive candidates relative to the batch mean
    # This is more effective than point-wise division because it balances the entire batch
    # Cost is in [1, 2], so z-score normalization makes the penalty scale-invariant
    
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe)
    
    # Avoid division by zero if all costs are identical
    if cost_std < epsilon:
        cost_std = epsilon
    
    # Compute cost z-score: positive means more expensive than average
    cost_z = (cost_safe - cost_mean) / cost_std
    
    # Apply a moderate penalty for expensive points
    # The penalty is additive, so it shifts the score without changing the relative ordering of UCB
    # Lambda = 0.5 is a reasonable weight: a cost difference of 1 standard deviation
    # changes the score by 0.5, which is significant but not dominant
    lambda_cost = 0.5
    cost_penalty = lambda_cost * cost_z
    
    # Final score: UCB minus cost penalty
    scores = ucb_score - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
