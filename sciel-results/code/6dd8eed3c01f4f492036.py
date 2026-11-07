# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 280  # Safety margin before 300s limit
_TIMEOUT_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a calibrated exploitation term plus relative uncertainty,
    weighted by inverse cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -50, 50)
    
    # Probability of improvement (PI)
    pi = norm.cdf(z_clipped)
    
    # Expected improvement (EI)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Calibrated exploitation: negative mean per unit cost
    # This directly rewards low predicted values
    exploitation = -mu / cost
    
    # Relative uncertainty term: encourages exploration in uncertain regions
    # but scaled relative to the scale of the problem
    sigma_scale = np.std(mu) + 1e-10
    exploration = (sigma / sigma_scale) / cost
    
    # Combine: exploitation dominates, exploration provides diversity
    # The blend is structural: exploitation term is primary signal,
    # exploration is secondary to avoid getting stuck
    scores = exploitation + 0.3 * exploration + 0.1 * (ei / cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > _TIMEOUT_MARGIN:
        # If we're taking too long, return what we have
        return scores
    
    return scores
# EVOLVE-BLOCK-END
