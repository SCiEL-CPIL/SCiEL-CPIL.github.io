# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    Uses Cost-Adjusted Expected Improvement (EI).
    
    The standard EI for minimization is:
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    We adjust for cost by dividing by the cost of evaluation.
    cost(x) = 1 + mean(x_scaled). Since we don't have x, we are given 'cost' directly.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    elapsed = _get_elapsed()
    if elapsed > 325: # Safety margin
         # Return a simple heuristic if we are out of time
         # Prefer low mu, low cost
         return (-mu / np.maximum(cost, 1e-9))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety clamps
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    # Improvement is defined as f_best - f(x). We want this to be positive.
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF of standard normal
    # phi is PDF of standard normal
    
    # Use scipy.stats.norm for numerical stability
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    
    # Handle case where sigma is very small to avoid numerical issues in division, 
    # though we already clamped sigma.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
