import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        sigma_safe = np.clip(sigma, 1e-9, None)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma for numerical stability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Cost-aware UCB acquisition:
    # UCB explores via mu + kappa*sigma, then we apply cost penalty
    # For minimization, lower is better, so we want to maximize (f_best - mu) + kappa*sigma
    # This is equivalent to EI-like but with linear sigma term instead of nonlinear
    
    # Use a kappa that balances exploration
    kappa = 1.5
    
    # UCB score (higher = more promising)
    ucb = (f_best - mu) + kappa * sigma
    
    # Ensure non-negative for meaningful penalty application
    ucb = np.maximum(ucb, 0.0)
    
    # Apply exponential cost penalty (stronger than current, matching best inspiration)
    penalty = np.exp(-1.0 * (cost - 1.0))
    
    scores = ucb * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
