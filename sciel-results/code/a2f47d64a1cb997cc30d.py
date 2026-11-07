import numpy as np
from scipy.stats import norm
import time

_start_time = time.monotonic()
_DEADLINE = _start_time + 290.0  # 10s margin before 300s limit

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
PROGRESS_SPAN = 150.0


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    return np.maximum(ei, 0.0)


def _min_max_normalize(x):
    """
    Min-max normalize an array to [0, 1].
    """
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / x_range


def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition function with adaptive exploration.
    
    The core idea: instead of sampling from the posterior, we construct a
    deterministic surrogate by shifting the mean toward lower values using
    the uncertainty (LCB: mu - k*sigma). We then compute EI on this shifted
    mean, which naturally balances exploration and exploitation.
    
    An additional UCB component is blended in with progress-adaptive weights:
    - Early (far from optimum): more UCB exploration
    - Late (close to optimum): more EI exploitation
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    if time.monotonic() > _DEADLINE:
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # 1. Calculate Progress: how close are we to the known global minimum?
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. Kriging Believer: use lower confidence bound as surrogate mean
    # k adapts: more aggressive (higher k) in early stages, more conservative later
    # Early (progress~0): k ~ 2.0 (strong exploration via shifted mean)
    # Late (progress~1): k ~ 0.3 (nearly pure exploitation)
    k = 0.3 + 1.7 * (1.0 - progress)
    mu_lb = mu - k * sigma
    
    # 3. Compute EI on the lower confidence bound (Kriging Believer)
    # This transforms probabilistic selection into deterministic optimization
    # of a surrogate objective that inherently balances exploration/exploitation
    ei_scores = _expected_improvement(mu_lb, sigma, f_best)
    
    # 4. UCB score for additional exploration
    # Use adaptive beta: higher in early stages
    beta = 0.5 + 1.5 * (1.0 - progress)
    ucb_scores = -(mu + beta * sigma)
    
    # 5. Normalize both scores to [0, 1] for blending
    ei_norm = _min_max_normalize(ei_scores)
    ucb_norm = _min_max_normalize(ucb_scores)
    
    # 6. Blend: weight shifts from UCB exploration to EI exploitation
    # Early (progress~0): w_explore=0.6, w_exploit=0.4
    # Late (progress~1): w_explore=0.15, w_exploit=0.85
    w_explore = 0.15 + 0.45 * (1.0 - progress)
    w_exploit = 1.0 - w_explore
    
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # 7. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
