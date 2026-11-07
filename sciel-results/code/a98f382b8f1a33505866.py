"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a Kriging Believer approach: evaluates EI at a shifted mean (mu - kappa*sigma)
to balance exploitation and exploration in a single coherent metric.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: The evaluator kills at 330s. We track time to ensure we return safely.
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0  # Seconds
_MARGIN = 5.0        # Safety margin


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
    Uses a Kriging Believer strategy: shift the posterior mean by kappa*sigma
    to account for exploration, then compute EI on the shifted mean.
    This creates a smooth trade-off between exploitation (low mu) and exploration (high sigma).
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

    # Kriging Believer: shift the mean by kappa * sigma
    # A larger kappa encourages exploration; smaller kappa favors exploitation.
    # For Styblinski-Tang 1D (multimodal), we use a moderate kappa.
    kappa = 1.0
    
    # Compute the shifted mean (lower confidence bound)
    mu_shifted = mu - kappa * sigma
    
    # Compute EI on the shifted mean, but use the original sigma for the uncertainty term
    # This is the standard Kriging Believer formulation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu_shifted) / sigma_safe
    z = np.clip(z, -10, 10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI on shifted mean: (f_best - mu_shifted) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu_shifted) * Phi_z + sigma_safe * phi_z
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
