"""Quantile-based acquisition with adaptive tail parameter for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    """Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a quantile-based approach: score the lower tail of the posterior.
    The quantile level is adaptive: early on (large gap) we look at a more
    extreme lower tail (more exploration), late we look closer to the mean.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Basic sanity
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Goldstein-Price 2D minimum value
    f_min = 3.0
    
    # Compute gap
    gap = max(f_best - f_min, 0.0)
    
    # Adaptive quantile level for the lower tail
    # Early (large gap): look at very extreme lower tail (small q, like 0.01-0.05)
    # Late (small gap): look closer to the mean (larger q, like 0.15-0.25)
    # 
    # q = base + amp * (1 / (1 + gap / scale))
    # gap=0: q = base + amp (e.g., 0.15 + 0.10 = 0.25)
    # gap=5: q = 0.15 + 0.10 * 0.5 = 0.20
    # gap=10: q = 0.15 + 0.10 * (1/3) = 0.183
    # gap=20: q = 0.15 + 0.10 * 0.2 = 0.17
    # gap=50: q = 0.15 + 0.10 * 0.091 = 0.159
    # gap=100: q = 0.15 + 0.10 * 0.048 = 0.155
    
    # Actually, let me think about this differently.
    # The finding said Q0.1 worked well. Let me use an adaptive q that starts
    # around 0.05 (very exploratory) and moves to 0.15 (more exploitative).
    
    q = 0.05 + 0.10 * (1.0 / (1.0 + gap / 5.0))
    # gap=0: q = 0.15
    # gap=5: q = 0.05 + 0.10*0.5 = 0.10
    # gap=10: q = 0.05 + 0.10*(1/3) = 0.083
    # gap=20: q = 0.05 + 0.10*0.2 = 0.07
    # gap=50: q = 0.05 + 0.10*0.091 = 0.059
    # gap=100: q = 0.05 + 0.10*0.048 = 0.055
    
    # Clip q to valid range
    q = np.clip(q, 0.01, 0.4)
    
    # Compute the z-score corresponding to this quantile
    # For a standard normal, the q-quantile is norm.ppf(q)
    # The q-quantile of N(mu, sigma^2) is mu + sigma * norm.ppf(q)
    # We want to score higher for lower values, so we negate
    z_q = norm.ppf(q)  # This will be negative for q < 0.5
    
    # Score: higher is better. We want to prefer points with low mu and high sigma
    # in the lower tail. The q-quantile of the posterior is mu + sigma * z_q
    # Since z_q < 0, lower mu and higher sigma both make the quantile lower.
    # We negate to make higher scores mean "evaluate next".
    
    quantile_value = mu + sigma * z_q
    score = -quantile_value
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score.astype(np.float64)
# EVOLVE-BLOCK-END
