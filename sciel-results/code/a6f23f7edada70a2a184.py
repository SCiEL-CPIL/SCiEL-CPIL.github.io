import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a pure UCB with high exploration coefficient to aggressively
    discover Branin's distant local minima.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Fall back to the fastest reasonable heuristic
        return -mu + 5.0 * sigma
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is finite and non-negative
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Pure UCB with high exploration coefficient
    # beta = 5.0 aggressively prioritizes regions with high posterior uncertainty,
    # which is crucial for discovering Branin's three distinct global minima
    beta = 5.0
    score = -mu + beta * sigma
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
