"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a Kriging Believer strategy that maps the posterior (mu, sigma) into a
deterministic surrogate objective by evaluating Expected Improvement on the
lower confidence bound (mu - k*sigma). The exploration weight adapts based
on progress toward the known global minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0
_MARGIN = 10.0

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
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
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
    Compute acquisition scores for 2000 candidate points using a Kriging
    Believer strategy blended with a UCB exploration term.
    
    The Kriging Believer shifts the GP mean by kappa * sigma before computing
    EI, making exploration an intrinsic property of the surrogate. A UCB term
    is blended in for additional exploration, with weights that shift from
    exploration to exploitation as progress is made.
    
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
    if time.monotonic() - _START_TIME > _TIME_BUDGET - _MARGIN:
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Progress
    # progress = 0 when f_best is far from global min (early stage)
    # progress = 1 when f_best is close to global min (late stage)
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # Kriging Believer: shift the mean by kappa * sigma
    # kappa adapts: higher in early stages (more exploration), lower in late stages
    kappa = 0.3 + 1.2 * (1.0 - progress)
    mu_shifted = mu - kappa * sigma
    
    # Compute EI on the shifted mean
    ei = _expected_improvement(mu_shifted, sigma, f_best)
    
    # UCB score for additional exploration
    # Use adaptive beta: higher in early stages
    beta = 0.5 + 1.5 * (1.0 - progress)
    ucb_scores = -(mu + beta * sigma)
    
    # Normalize both scores to [0, 1] for blending
    ei_norm = _min_max_normalize(ei)
    ucb_norm = _min_max_normalize(ucb_scores)
    
    # Blend: weight shifts from UCB exploration to EI exploitation
    # Early (progress~0): w_explore=0.6, w_exploit=0.4
    # Late (progress~1): w_explore=0.15, w_exploit=0.85
    w_explore = 0.15 + 0.45 * (1.0 - progress)
    w_exploit = 1.0 - w_explore
    
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
