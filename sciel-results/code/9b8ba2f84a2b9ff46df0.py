"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement normalized by cost.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        return 1.0 / (np.maximum(cost, 1e-9))
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI is defined for minimization.
    # Improvement is f_best - f(x).
    # E[Improvement] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in norm.pdf/cdf for extreme values
    Z = (f_best - mu) / sigma_safe
    Z_clipped = np.clip(Z, -10, 10)
    
    # Calculate EI components
    EI = (f_best - mu) * norm.cdf(Z_clipped) + sigma_safe * norm.pdf(Z_clipped)
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware scoring
    # Structural Change: Use multiplicative cost adjustment (divide by cost)
    # This prioritizes points with high expected improvement per unit cost.
    # Cost is in [1, 2].
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = EI / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
