import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0
_MARGIN = 2.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D (minimization).
    Uses Expected Improvement (EI) with a small exploration bonus.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _MARGIN:
        return -mu

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Small exploration bonus to help escape local optima
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_safe
    
    scores = ei + exploration
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
