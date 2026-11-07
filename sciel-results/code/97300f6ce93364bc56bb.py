import numpy as np
import time
from scipy.stats import norm

# Global start time for the deadline guard (set on first call).
_START_TIME = None
# Hard wall-clock limit (seconds) for the whole evaluation.
_HARD_LIMIT = 330.0
# Safety margin so we return well before the limit.
_MARGIN = 5.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Combines Expected Improvement (EI) for exploitation with a UCB term
    for exploration. For minimization, we want low mu and high sigma.
    
    EI for minimization:
        EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
        where z = (f_best - mu(x)) / sigma(x)
    
    UCB for minimization:
        UCB(x) = -mu(x) + beta * sigma(x)
    
    We use a weighted combination: score = w_ei * EI + w_ucb * UCB
    """
    elapsed = _elapsed()
    if elapsed > _HARD_LIMIT - _MARGIN:
        # Deadline exceeded, return a simple score
        scores = -mu + sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases where sigma is very small
    sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
    
    # Compute Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) = norm.cdf(z), phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # UCB for minimization: -mu + beta * sigma
    # Increase beta to encourage exploration, similar to successful pure UCB variants
    beta = 2.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with tuned weights
    # Shift weight towards UCB to leverage the successful exploration behavior
    # EI provides local refinement, UCB provides global coverage
    w_ei = 0.2
    w_ucb = 1.0
    
    scores = w_ei * ei + w_ucb * ucb
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
