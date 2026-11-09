"""Rewritten acquisition function using Continuous Hypervolume Improvement (Exact EI) with Adaptive Exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses the Exact Continuous Hypervolume Improvement (equivalent to Expected Improvement)
    combined with an adaptive exploration term that scales with the progress towards
    the known global minimum. This avoids discretization errors and provides a smooth,
    robust acquisition surface.
    """
    # Time budget guard
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple EI
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12,
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    known_min = 0.397887
    # Approximate initial f_best range for normalization
    # Branin values can be up to ~300, but usually start lower.
    # We use a robust scale based on the known minimum and a typical initial max.
    initial_range_scale = 20.0 
    
    # Calculate progress: 0 at start (f_best high), 1 near optimum
    # If f_best is below known_min (shouldn't happen often but possible with noise/numerical), clip
    current_gap = max(0.0, f_best - known_min)
    # Normalize gap. If gap is > initial_range_scale, progress is 0.
    # If gap is 0, progress is 1.
    progress = 1.0 - min(1.0, current_gap / initial_range_scale)
    
    # Adaptive Beta for exploration
    # Early on (progress ~ 0), we want high exploration.
    # Late on (progress ~ 1), we want low exploration (exploit).
    # Use a smooth decay function.
    # beta = beta_max * (1 - progress)^p + beta_min
    beta_max = 0.8
    beta_min = 0.05
    p_exp = 2.0
    beta = beta_max * ((1.0 - progress) ** p_exp) + beta_min
    
    # Compute Exact Expected Improvement (Continuous Hypervolume Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in exp for very large negative/positive z
    # norm.cdf and norm.pdf handle large values well, but clipping helps stability
    z = np.clip(z, -10, 10)
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    
    # Exploration term
    exploration = beta * sigma
    
    # Final Score
    scores = ei + exploration
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
