"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) with posterior means in objective units
    - sigma: array of shape (2000,) with posterior standard deviations in objective units
    - f_best: lowest observed value so far
    
    Returns:
    - array of shape (2000,) with scores; higher means more promising
    """
    # Check deadline - if we're close to timeout, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: return negative mu (pure exploitation) to finish quickly
        return -mu
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Use stable computation to avoid overflow/underflow
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma_safe * pdf
    
    # UCB term for exploration
    # Adaptive beta: smooth decay based on f_best
    # As f_best decreases (getting closer to optimum), beta decreases (more exploitation)
    # As f_best is large (far from optimum), beta increases (more exploration)
    # log1p(0) = 0, so beta starts around 1.0
    # log1p(100) ~ 4.6, beta ~ 1.46
    # log1p(10000) ~ 9.2, beta ~ 1.92
    beta = 1.0 + 0.1 * np.log1p(max(f_best, 0))
    
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # Using a weighted sum. Since both are in objective units, we can sum them directly.
    # EI is generally a stronger signal for minimization.
    # We use 0.6 EI and 0.4 UCB to balance.
    score = 0.6 * ei + 0.4 * ucb
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
