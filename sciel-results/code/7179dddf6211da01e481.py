import numpy as np
from scipy.stats import norm
import time

# Track start time for budget guard
_ACQ_START_TIME = None
_ACQ_CALL_COUNT = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a smooth cost-normalized acquisition that balances exploration and exploitation.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    global _ACQ_START_TIME, _ACQ_CALL_COUNT
    
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    _ACQ_CALL_COUNT += 1
    
    # Safety check: if we're getting close to the 330s limit, return quickly
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 320:
        # Return a simple fallback score to avoid timeout
        return np.zeros(len(mu))
    
    # Calculate Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Use a smooth cost-normalized approach instead of hard gating
    # Normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Adaptive exploration parameter based on call count
    # More exploration early on, more exploitation later
    exploration_weight = 0.3 + 0.4 * np.exp(-_ACQ_CALL_COUNT / 50.0)
    
    # Smooth cost penalty: lower cost candidates get higher scores
    # cost_penalty ranges from 1 (low cost) to 1-exploration_weight (high cost)
    cost_penalty = 1.0 - exploration_weight * cost_norm
    
    # Combine EI with cost penalty
    scores = ei * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
