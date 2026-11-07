import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded(margin=15.0):
    """Check if we're close to the 330s deadline."""
    return _get_elapsed() > (330.0 - margin)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement
    with an exploration term, designed for Ackley 2D optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP std deviations in objective units
        f_best: lowest observed value (scalar)
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return simple EI
    if _is_deadline_exceeded():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-8)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    
    # Ensure cost is positive and finite
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-8)
    
    # Compute z-scores for Expected Improvement
    delta = f_best - mu
    z = delta / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    ei = delta * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration term: encourage exploration in high-uncertainty regions
    # Use a tunable exploration weight
    exploration_weight = 0.3
    exploration = exploration_weight * sigma
    
    # Combine EI and exploration
    combined = ei + exploration
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # Use a power less than 1 to soften the cost penalty
    scores = combined / np.power(cost, 0.5)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If all scores are non-positive, add a small random tie-breaker
    if np.all(scores <= 0):
        scores = scores + np.random.uniform(0, 1e-6, size=scores.shape)
    
    return scores
