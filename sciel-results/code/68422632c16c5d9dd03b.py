"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a Kriging Believer strategy that maps the posterior (mu, sigma) into a
deterministic surrogate objective by evaluating Expected Improvement on the
lower confidence bound (mu - k*sigma), blended with a UCB-like exploration term.
The exploration weight adapts based on progress toward the known global minimum.
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


def _probability_of_improvement(mu, sigma, f_best):
    """
    Calculate Probability of Improvement for minimization.
    POI = Phi((f_best - mu) / sigma)
    This gives the probability that a candidate point yields a value lower than f_best.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    return np.clip(poi, 0.0, 1.0)


def _poi_weighted_score(mu, sigma, f_best, progress):
    """
    POI-based acquisition that weights the probability of improvement by the expected gain.
    
    The score combines:
    1. POI: probability of beating f_best (exploration-friendly)
    2. Expected gain: (f_best - mu) when mu < f_best (exploitation)
    3. Uncertainty bonus: sigma * POI (explores uncertain regions)
    
    As progress increases, we shift weight from uncertainty to expected gain.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    poi = np.clip(poi, 0.0, 1.0)
    
    # Expected improvement (magnitude of gain)
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Uncertainty-weighted POI: explores regions with high sigma AND high chance of improvement
    uncertainty_bonus = sigma_safe * poi
    
    # Normalize components
    poi_norm = _min_max_normalize(poi)
    ei_norm = _min_max_normalize(ei)
    unc_norm = _min_max_normalize(uncertainty_bonus)
    
    # Adaptive weights: early stages favor uncertainty, late stages favor exploitation
    # Early (progress~0): w_poi=0.3, w_ei=0.2, w_unc=0.5
    # Late (progress~1): w_poi=0.2, w_ei=0.6, w_unc=0.2
    w_poi = 0.2 + 0.1 * (1.0 - progress)
    w_ei = 0.2 + 0.4 * progress
    w_unc = 1.0 - w_poi - w_ei
    
    scores = w_poi * poi_norm + w_ei * ei_norm + w_unc * unc_norm
    return scores


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
    Compute acquisition scores for 2000 candidate points.
    
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

    # 1. Calculate Progress
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. POI-based acquisition with adaptive weighting
    # This approach naturally balances exploration and exploitation:
    # - POI favors regions with high probability of improvement
    # - EI favors regions with large expected gains
    # - Uncertainty bonus explores regions with high sigma
    scores = _poi_weighted_score(mu, sigma, f_best, progress)
    
    # 7. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
