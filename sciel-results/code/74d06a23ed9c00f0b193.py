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
    Cost-aware acquisition function using Expected Improvement (EI).
    
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
    
    # Calculate Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost Normalization
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
