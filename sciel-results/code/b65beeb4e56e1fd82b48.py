"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: we must return within the 300s limit per evaluation.
_DEADLINE = 300.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior
    sigma : ndarray, shape (2000,)
        Std of GP posterior
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check time budget
    if _time_left() < 1.0:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)
    
    # Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    POI = norm.cdf(z)
    
    # Ensure POI is in [0, 1] and finite
    POI = np.clip(POI, 0.0, 1.0)
    POI = np.where(np.isfinite(POI), POI, 0.0)
    
    # Dynamic Exploration Factor
    # Derived from global maximum uncertainty
    max_sigma = np.max(sigma)
    if max_sigma < 1e-9:
        # If uncertainty is negligible, pure exploitation (POI) is fine, 
        # but we still divide by cost.
        exploration_weight = np.ones_like(POI)
    else:
        # Relative uncertainty: 0 to 1
        rel_uncertainty = sigma / max_sigma
        # Weight: 1.0 to 1.5 (moderate exploration boost for high uncertainty points)
        # This encourages exploring regions with high sigma if they also have decent POI
        exploration_weight = 1.0 + 0.5 * rel_uncertainty
        
    # Cost-aware Score
    # Higher POI and lower cost are better.
    # The exploration weight boosts high-uncertainty candidates slightly.
    scores = (POI * exploration_weight) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
