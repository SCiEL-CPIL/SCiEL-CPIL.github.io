"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Analytic Expected Improvement (EI).
    
    This is a structural change from the previous Monte Carlo Max-Sampling approach.
    EI is a deterministic, analytically derived acquisition function that balances
    exploitation (low mu) and exploration (high sigma) based on the probability
    of improving upon the current best value f_best.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu) with tiny exploration
        scores = -mu + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Calculate Expected Improvement
    # For minimization:
    # Improvement = max(f_best - y, 0)
    # E[Improvement] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma, Phi is CDF, phi is PDF of standard normal.
    
    # Handle division by zero/small sigma
    Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in CDF/PDF for extreme values, 
    # though scipy handles it well. 
    # If Z is very large positive, Phi(Z) ~ 1, phi(Z) ~ 0 -> EI ~ f_best - mu
    # If Z is very large negative, Phi(Z) ~ 0, phi(Z) ~ 0 -> EI ~ 0
    
    cdf_Z = norm.cdf(Z)
    pdf_Z = norm.pdf(Z)
    
    scores = (f_best - mu) * cdf_Z + sigma * pdf_Z
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
