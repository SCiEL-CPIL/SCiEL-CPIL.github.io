"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track elapsed time to ensure we return well before the 330s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin


def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid EI + UCB approach:
    - EI for exploitation (points likely to improve over f_best)
    - UCB for exploration (points with high uncertainty)
    - Small sigma bonus to encourage exploration in early stages
    
    For minimization: higher score means more likely to be evaluated next.
    """
    # Deadline check: if we're too close to the limit, return a simple score
    if _check_deadline():
        return -mu + 2.0 * sigma

    # Expected Improvement (EI) component
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB (Upper Confidence Bound) component for exploration
    # Use a beta parameter that balances exploration and exploitation
    beta = 2.0

    # Hybrid: blend EI and UCB
    # For minimization:
    #   EI: higher when mu is much lower than f_best (good)
    #   UCB for minimization: we want to explore where mu - beta*sigma is low, so score = -mu + beta*sigma
    
    # Let's use: score = EI + alpha * (-mu + beta * sigma)
    # This combines exploitation (EI) with exploration (prefer low mu, high sigma)
    
    alpha = 0.5  # weight for UCB exploration term
    score = ei + alpha * (-mu + beta * sigma_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
