import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to ensure we finish before 330s
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: Scalar, the best objective value observed so far.
        cost: (2000,) array of evaluation costs (1 + mean(x)).
        
    Returns:
        (2000,) array of acquisition scores.
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    # Convert to numpy arrays for consistency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Calculate z-score for EI
    z = (f_best - mu) / safe_sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Clip negative EI to zero (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring with milder penalty
    # Linear cost penalty (alpha=1.0) lets EI signal dominate more
    # This helps find actual improvements rather than just cheap points
    alpha = 1.0
    cost_penalty = np.power(safe_cost, alpha)
    score = ei / cost_penalty
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the 6D landscape
    # Scale exploration to be ~10% of typical EI values
    max_sigma = np.max(safe_sigma)
    if max_sigma > 0:
        exploration_bonus = 0.1 * safe_sigma / max_sigma
        score = score + exploration_bonus
    
    # Ensure output is finite and real
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
