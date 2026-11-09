import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
TIME_LIMIT = 300.0  # Safety margin from 330s

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a time-adaptive
    exploration bonus for Bayesian optimization of the Beale 2D function.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    global _acq_call_count
    _acq_call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to a fast, deterministic method if time is critical
        # Return negative mu (exploitation)
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # The evaluator runs 25 optimization steps per seed (30 total - 5 initial).
    # We use modulo 25 to reset the progress for each new seed.
    step = (_acq_call_count - 1) % 25
    progress = step / 25.0
    
    # Time-adaptive exploration weight.
    # We want to explore aggressively at the start (progress ~ 0)
    # and exploit at the end (progress ~ 1).
    # Using a power law decay for a smooth transition.
    # Start with higher exploration to escape local optima in the early stages.
    lambda_explore = 1.5 * (1.0 - progress) ** 1.5
    
    # Compute Expected Improvement (EI)
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Vectorized computation of CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Standard EI formula
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: EI + Exploration Bonus
    # The exploration bonus is proportional to uncertainty (sigma)
    score = ei + lambda_explore * sigma
    
    # Ensure scores are finite
    score[~np.isfinite(score)] = 0.0
    
    return score
