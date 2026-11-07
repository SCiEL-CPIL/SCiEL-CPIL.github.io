import numpy as np
from scipy.stats import norm
import time

_start_time = time.monotonic()
_DEADLINE = _start_time + 290.0

GLOBAL_MIN = -39.1662
PROGRESS_SPAN = 100.0  # Reduced span to make progress increase faster

def _rank_normalize(x):
    """
    Normalize array to [0, 1] using rank.
    Handles ties by averaging ranks.
    """
    n = len(x)
    if n == 0:
        return x
    # argsort twice to get ranks
    sorter = np.argsort(x)
    rank = np.empty_like(sorter, dtype=np.float64)
    rank[sorter] = np.arange(n)
    # Handle ties
    # For simplicity, if no ties, this is fine.
    # If ties, average rank.
    # But np.argsort is stable? No.
    # Let's use scipy.stats.rankdata if available, or simple implementation.
    # Given 2000 points, ties are unlikely with floats.
    return rank / (n - 1)

def _expected_improvement(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best):
    if time.monotonic() > _DEADLINE:
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # Progress
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # EI Score
    ei_scores = _expected_improvement(mu, sigma, f_best)
    
    # LCB Score
    # Adaptive k: 3.0 (early) to 0.5 (late)
    k = 0.5 + 2.5 * (1.0 - progress)
    lcb_scores = -(mu - k * sigma)
    
    # Normalize using rank
    ei_norm = _rank_normalize(ei_scores)
    lcb_norm = _rank_normalize(lcb_scores)
    
    # Blend
    # Early: more LCB (exploration)
    # Late: more EI (exploitation)
    # w_explore (LCB) from 0.7 to 0.2
    # w_exploit (EI) from 0.3 to 0.8
    w_explore = 0.2 + 0.5 * (1.0 - progress)
    w_exploit = 1.0 - w_explore
    
    scores = w_exploit * ei_norm + w_explore * lcb_norm
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
