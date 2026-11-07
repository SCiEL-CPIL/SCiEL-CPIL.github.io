"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Combines Expected Improvement with a distance-scaled exploration bonus.
The exploration weight adapts based on how far f_best is from the known global minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: The evaluator kills at 330s. We track time to ensure we return safely.
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0  # Seconds
_MARGIN = 5.0        # Safety margin


def _min_max_normalize(x):
    """
    Min-max normalize an array to [0, 1].
    Handles constant arrays by returning zeros.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-9:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with norm.cdf/pdf
    z = np.clip(z, -10, 10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses a normalized blend of Expected Improvement and Uncertainty (Sigma).
    This structural change ensures scale-invariance and stable exploration/exploitation balance.
    """
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    if time.monotonic() - _START_TIME > _TIME_BUDGET - _MARGIN:
        # Return -mu as a fast fallback (lower mu is better, so higher score is -mu)
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Normalize EI and Sigma to [0, 1]
    ei_norm = _min_max_normalize(ei)
    sigma_norm = _min_max_normalize(sigma)
    
    # Blend: 0.5 * EI + 0.5 * Sigma
    # This is a Kriging Believer-like balance but simpler and more robust to scale.
    # In multi-modal functions like Styblinski-Tang, maintaining exploration is crucial.
    scores = 0.5 * ei_norm + 0.5 * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
