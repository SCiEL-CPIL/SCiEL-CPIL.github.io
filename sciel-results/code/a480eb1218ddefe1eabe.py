import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s


def _check_deadline():
    """Return True if we should abort and return a fallback immediately."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) for minimization,
    with a small exploration floor to avoid collapsing too early.
    
    mu, sigma : (2000,) arrays, sigma is the GP posterior std dev
    f_best : lowest observed value so far
    Returns: (2000,) scores, higher = more promising
    """
    # Handle edge cases
    if sigma is None or len(mu) != len(sigma):
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Check deadline
    if _check_deadline():
        # Fallback: pure exploitation
        return -mu
    
    n = len(mu)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute the standardized improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, we want high EI scores
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a small exploration term to prevent pure exploitation
    # This helps escape local optima early on
    # Use a small constant multiplier for sigma
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
