"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with robust numerical handling.
    
    This combines the pure EI strategy of the baseline (which scored 1.0) with the
    robust input validation and sigma clipping found in higher-performing inspirations
    that attempted to mix strategies. The robustness ensures stability without
    introducing the exploration bias that degraded performance in previous attempts.
    
    mu: (N,) posterior mean in objective units
    sigma: (N,) posterior std in objective units
    f_best: best observed value so far in objective units
    
    Returns: (N,) scores, higher is better.
    """
    # Ensure inputs are numpy arrays of float64 for stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Clip sigma to a small positive value to avoid division by zero and numerical issues
    # This is a refinement from Inspiration 1, which showed that robust handling is important.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate standard normal variable z
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using norm.cdf and norm.pdf from scipy.stats
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite. If any calculation results in NaN or Inf, set to 0.
    # This prevents invalid results from propagating.
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
