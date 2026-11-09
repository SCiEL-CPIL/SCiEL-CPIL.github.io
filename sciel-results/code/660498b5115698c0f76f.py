"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # seconds, with safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and UCB.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coords)
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Ensure finite inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Clip to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid overflow in norm.cdf/pdf
    z = np.clip(z, -100, 100)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB component for exploration
    # Adaptive beta: increase exploration early, decrease later
    # Use a simple heuristic based on f_best magnitude
    beta = 1.2
    ucb = -mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Blend EI and UCB
    # Weight toward EI for exploitation, UCB for exploration
    alpha = 0.6
    score = (alpha * ei + (1 - alpha) * ucb) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
