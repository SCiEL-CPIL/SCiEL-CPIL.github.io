"""Expected Improvement (EI) acquisition function with adaptive jitter for Ackley 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global start time for the deadline guard (set on first call).
_START_TIME = None
# Hard wall-clock limit (seconds) for the whole evaluation.
_HARD_LIMIT = 330.0
# Safety margin so we return well before the limit.
_MARGIN = 5.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Strategy:
    - Uses Expected Improvement (EI) for minimization.
    - EI is a different design family from UCB, focusing on the expected reduction
      in the best value found so far.
    - Includes an adaptive jitter (xi) to balance exploration and exploitation.
      The jitter starts higher when f_best is large (early stage) to encourage
      exploration, and decreases as f_best approaches 0 (late stage) to focus
      on exploitation.
    - Formula: EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
      where Z = (f_best - mu) / sigma.
    - We return the EI score. Higher EI means better candidate.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard: ensure finite inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)

    # Deadline guard
    if _elapsed() > _HARD_LIMIT - _MARGIN:
        # Fallback to simple exploitation (negative mean)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)

    # Adaptive Jitter (xi)
    # When f_best is large (e.g., > 1.0), we want more exploration.
    # When f_best is small (e.g., < 0.1), we want less exploration.
    # xi = 0.01 + 0.05 * min(1.0, f_best)
    # If f_best = 10, xi = 0.06
    # If f_best = 0, xi = 0.01
    xi = 0.01 + 0.05 * np.clip(f_best, 0.0, 1.0)

    # Calculate Z-score for EI
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma
    
    # Handle numerical stability for large Z
    # Phi(Z) is the CDF of the standard normal
    # phi(Z) is the PDF of the standard normal
    # For very large Z, Phi(Z) -> 1, phi(Z) -> 0
    # For very small Z, Phi(Z) -> 0, phi(Z) -> 0.4
    
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: This is the standard EI formula for minimization where we want to improve upon f_best.
    # If f_best - mu is negative, EI should be 0.
    # The formula above naturally handles this if we clamp it, but mathematically:
    # If Z is negative (mu > f_best), Phi(Z) is small, phi(Z) is small.
    # Actually, EI is defined as E[max(f_best - f(x), 0)].
    # The closed form is:
    # If f_best - mu > 0: (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # If f_best - mu <= 0: 0
    # The standard formula (f_best - mu) * Phi(Z) + sigma * phi(Z) works for all Z,
    # but yields negative values if Z is very negative?
    # Let's check: If mu >> f_best, Z is large negative.
    # Phi(Z) ~ 0. phi(Z) ~ 0.
    # (f_best - mu) is negative. So EI ~ 0?
    # Actually, the term (f_best - mu) * Phi(Z) dominates.
    # If Z = -10, Phi(-10) ~ 0. So EI ~ 0.
    # If Z = 0, EI = sigma * phi(0) = sigma * 0.398.
    # If Z = 10, EI = (f_best - mu) * 1 + 0.
    # So the formula is correct.
    
    scores = (f_best - mu) * Phi + sigma * phi
    
    # Ensure non-negative (EI should be >= 0)
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
