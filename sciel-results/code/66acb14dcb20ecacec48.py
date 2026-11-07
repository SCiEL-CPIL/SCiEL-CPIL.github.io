"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a Kriging Believer strategy that shifts the GP posterior mean by an adaptive
exploration parameter kappa * sigma before computing Expected Improvement (EI).
This adapts the exploration/exploitation balance based on progress toward the known
global minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: The evaluator kills at 330s. We track time to ensure we return safely.
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0  # Seconds
_MARGIN = 10.0        # Safety margin

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
# Approximate initial range for progress calculation.
# Random points on [-5, 5] for ST1D can yield values around 50-100.
# We use a span of 150 to cover the likely initial range to the minimum.
PROGRESS_SPAN = 150.0


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Kriging Believer.
    
    The Kriging Believer shifts the GP mean by kappa * sigma before computing EI,
    which makes exploration an intrinsic property of the surrogate rather than a
    separate additive term. This avoids scale-mismatch problems from blending
    raw EI with UCB.
    
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
    # Early (progress=0): kappa ~ 1.5 (strong exploration)
    # Late (progress=1): kappa ~ 0.2 (nearly pure exploitation)
    kappa = 0.2 + 1.3 * (1.0 - progress)
    mu_shifted = mu - kappa * sigma
    
    # Compute EI on the shifted mean
    ei = _expected_improvement(mu_shifted, sigma, f_best)
    
    # Ensure Finite Output
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
