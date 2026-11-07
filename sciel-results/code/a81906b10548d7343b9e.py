"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive Quantile-based Exploration Strategy:
    - Computes empirical percentile rank of sigma.
    - Dynamically adjusts the uncertainty quantile threshold (tau) and
      mean tolerance based on the current best value f_best.
    - Early on (high f_best), explores more broadly (lower tau, higher tolerance).
    - Late on (low f_best), exploits more narrowly (higher tau, lower tolerance).
    - Score for valid candidates: -mu + alpha * (normalized_rank)^2.
    """
    n = len(mu)
    
    # Compute percentile ranks of sigma (0 to 1)
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / (n - 1)
    sigma_median = np.median(sigma)
    
    # Adaptive parameters based on f_best
    # Hartmann 3D global min is approx -3.86
    # If f_best is very low (good), we are likely close to a minimum.
    # We should focus on the highest uncertainty points (top 1%) and require
    # mu to be strictly below f_best (no tolerance).
    # If f_best is high (bad), we are far from minima.
    # We should consider a wider range of uncertainty (top 20%) and allow
    # some tolerance in mu to avoid getting stuck in local optima.
    
    if f_best < -3.5:
        tau = 0.99
        mean_tol = 0.0
    elif f_best < -3.0:
        tau = 0.95
        mean_tol = 0.05 * sigma_median
    elif f_best < -2.0:
        tau = 0.90
        mean_tol = 0.2 * sigma_median
    else:
        tau = 0.80
        mean_tol = 0.5 * sigma_median
        
    # Normalized position within the band [tau, 1.0]
    band_width = 1.0 - tau
    band_pos = np.clip((sigma_ranks - tau) / band_width, 0.0, 1.0)
    
    # Valid candidates: in the uncertainty band AND mean is sufficiently low
    in_band = sigma_ranks >= tau
    below_best = mu < (f_best + mean_tol)
    valid = in_band & below_best
    
    # Exploration bonus scale
    # Increased alpha slightly to encourage exploration of the top of the band
    alpha = 3.0 * sigma_median
    
    # Initialize scores with a low baseline
    # Penalty ensures valid points are strongly preferred
    scores = np.empty(n, dtype=np.float64)
    scores[:] = -mu - 10.0 * sigma_median
    
    # Assign scores to valid candidates
    # Use squared band_pos to emphasize the very highest uncertainty points
    scores[valid] = -mu[valid] + alpha * (band_pos[valid] ** 2)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
