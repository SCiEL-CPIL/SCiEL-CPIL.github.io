"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # 290s safety margin from 300s limit
        _DEADLINE = time.time() + 290
    return _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    with dynamic exploration factor.
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    deadline = _get_deadline()
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for POI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Dynamic exploration factor: adapt based on global uncertainty
    # When uncertainty is high, explore more; when low, exploit
    max_sigma = np.max(sigma_safe)
    if max_sigma < 1e-9:
        # Very low uncertainty - exploit strongly
        exploration_weight = 0.1
    else:
        # Scale exploration weight inversely with relative uncertainty
        # Higher global uncertainty -> higher exploration weight
        exploration_weight = min(1.0, max_sigma / (max_sigma + 0.1))
    
    # Exploration term: reward high-uncertainty regions
    # Normalize sigma to [0, 1] range for stable weighting
    sigma_norm = sigma_safe / (max_sigma + 1e-9)
    
    # Combined score: POI weighted by exploration + exploitation balance
    # exploitation: POI (probability of beating current best)
    # exploration: normalized uncertainty
    score = (1.0 - exploration_weight) * poi + exploration_weight * sigma_norm
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    score = score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
