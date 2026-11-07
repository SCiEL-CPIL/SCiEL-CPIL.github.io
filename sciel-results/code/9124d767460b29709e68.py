import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin below 300s per evaluation, well under 330s total


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI per unit cost
    # Add a small exploration term to encourage visiting uncertain regions
    # but weighted by cost efficiency
    exploration = sigma_safe / cost_safe
    
    # Combine: primary is cost-aware EI, with a small exploration bonus
    # The exploration helps escape local optima in the GP surrogate
    score = ei / cost_safe + 0.1 * exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
