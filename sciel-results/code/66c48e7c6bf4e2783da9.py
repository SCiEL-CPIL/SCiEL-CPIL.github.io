"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Hybrid Acquisition: Probability of Improvement (PoI) 
    with an Exploitation Bonus.
    
    This approach combines the robustness of PoI (which favors high confidence 
    of improvement, effective in multimodal landscapes) with a small 
    exploitation term (focusing on lower mean predictions) to guide the search 
    more precisely when uncertainty is low.
    
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
        # Fallback: return a simple cost-aware exploitation score
        # Prefer cheap points with low mean
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_safe, 1e-9)
        # Simple fallback: -mu / cost
        fallback = -mu_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Ensure inputs are arrays and handle types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure f_best is a scalar float
    if hasattr(f_best, 'min'):
        f_best_val = float(np.min(f_best))
    else:
        f_best_val = float(f_best)
        
    # Handle edge case where f_best is 0 or very small
    if f_best_val < 1e-6:
        f_best_val = 1e-6

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best_val - mu) / sigma_safe
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Calculate Exploitation Term
    # Raw improvement: how much better is the mean than f_best?
    improvement = np.maximum(0.0, f_best_val - mu)
    
    # Normalize improvement to [0, 1] relative to current best
    # This prevents the term from dominating when f_best is large
    norm_improvement = improvement / f_best_val
    
    # Combine PoI and Exploitation
    # PoI is the primary driver (robustness)
    # Norm_improvement is a secondary driver (precision)
    # Weight of 0.1 for exploitation is a small nudge
    hybrid_score = poi + 0.1 * norm_improvement
    
    # Cost-aware: divide by cost
    # Cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    scores = hybrid_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
