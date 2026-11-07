"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_BUDGET = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) acquisition function for minimization.
    
    For minimization, EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    This balances exploitation (low mu) and exploration (high sigma)
    in a theoretically principled way.
    """
    # Check time budget - if we're running low, fall back to simple UCB
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_BUDGET:
        # Emergency fallback: simple UCB
        beta = 1.0
        scores = -mu + beta * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        return scores
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    # Handle cases where sigma is very small (numerical stability)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI (should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
