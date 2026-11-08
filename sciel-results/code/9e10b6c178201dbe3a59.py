"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're getting close to the 330s limit, return a safe fallback
    if _elapsed() > 320:
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        # Fallback: exploit low mu, penalize high cost
        return -mu / np.maximum(cost, 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    
    # Use numerically stable computation
    # For very large positive z, Phi(z) ~ 1 and phi(z) ~ 0, so EI ~ improvement
    # For very negative z, EI ~ 0
    ei = np.zeros_like(mu)
    
    # Where z is reasonable
    mask = np.isfinite(z)
    z_safe = np.where(mask, z, 0.0)
    
    # Compute EI components
    cdf_vals = norm.cdf(z_safe)
    pdf_vals = norm.pdf(z_safe)
    
    ei[mask] = improvement[mask] * cdf_vals[mask] + sigma[mask] * pdf_vals[mask]
    
    # For very negative z (improvement very negative, i.e., mu >> f_best), EI should be ~0
    # For very positive z (improvement very positive, i.e., mu << f_best), EI ~ improvement
    
    # Clip to ensure non-negative EI
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    scores = ei / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
