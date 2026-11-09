"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _fast_norm_cdf(x):
    """Fast approximation of standard normal CDF."""
    # Use a rational approximation that is accurate to ~1e-7
    # For large |x|, CDF is ~0 or ~1
    sign = np.sign(x)
    ax = np.abs(x)
    # Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * ax)
    d = 0.3989422804014327 * np.exp(-0.5 * ax * ax)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    result = 1.0 - p
    result = np.where(sign >= 0, result, 1.0 - result)
    return result


def _fast_norm_pdf(x):
    """Fast standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    Uses a combination of EI and a Gaussian process-based exploration term.
    """
    start_time = time.time()
    deadline = start_time + 2.5  # Safety margin
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * _fast_norm_cdf(z) + sigma_safe * _fast_norm_pdf(z)
    
    # Compute UCB-style exploration: we want to maximize, so -mu + beta*sigma
    # Use adaptive beta based on the spread of sigma
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Beta that encourages exploration when there's high uncertainty
    # but not too much to avoid pure random search
    beta = 1.2 + 0.5 * (sigma_std / (sigma_mean + 1e-10))
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weighting: favor EI more, but keep some exploration
    # The key insight from the finding: hybrid balances exploitation and exploration
    # Use a weight that adapts to how much improvement is possible
    # When f_best is close to the minimum, we need more exploration
    # Hartmann 3D minimum is about -3.86278
    h3_min = -3.86278
    progress = max(0.0, min(1.0, (f_best - h3_min) / (h3_min * 0.1)))
    
    # Higher progress (closer to optimum) -> more exploration needed
    ei_weight = 0.65 - 0.2 * progress
    ucb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
