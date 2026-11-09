"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard normal variable for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB component: f_best - mu + beta * sigma
    # This gives positive scores to high-uncertainty regions even when mu > f_best
    beta = 2.0
    ucb = (f_best - mu) + beta * sigma_safe
    ucb = np.maximum(ucb, 0.0)
    
    # Blend EI and UCB: EI captures exploitation, UCB adds exploration
    # Use a weighted combination that emphasizes exploration more
    scores = 0.5 * ei + 0.5 * ucb
    
    # Cost-aware: divide by cost
    scores = scores / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
