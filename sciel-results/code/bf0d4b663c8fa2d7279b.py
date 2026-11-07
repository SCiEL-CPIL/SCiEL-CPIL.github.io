"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # 280s safety margin before 300s limit


def _check_deadline():
    """Return True if we should stop and return early."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value (lowest)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Check deadline
    if _check_deadline():
        # Return simple expected improvement as fallback
        z = (f_best - mu) / (sigma + 1e-12)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-12)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # UCB-like term for exploration
    # Balance exploitation (EI) with exploration (sigma)
    # Use a weighted combination that favors exploration early
    
    # Cost-normalized EI
    ei_cost = ei / cost
    
    # Exploration bonus: reward high uncertainty per unit cost
    # This helps find new promising regions
    exploration = sigma / cost
    
    # Combined score: primarily EI per cost, with exploration bonus
    # The exploration term is scaled down to not dominate
    score = ei_cost + 0.1 * exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
