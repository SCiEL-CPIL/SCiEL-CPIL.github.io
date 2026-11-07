"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Quantile-based exploration strategy:
    - Compute empirical percentile rank of sigma for each candidate (0 to 1).
    - Focus on the top 5% of uncertainty (rank >= 0.95).
    - Only consider candidates whose predicted mean is below f_best.
    - Score = -mu + alpha * (sigma_rank - 0.95) / 0.05 for valid candidates,
      where the normalized rank in [0,1] within the top-5% band provides a
      smooth exploration bonus. Invalid candidates get a low but finite score.
    """
    n = len(mu)
    
    # Compute percentile ranks of sigma (0 to 1, higher = more uncertain)
    # Use argsort of argsort for rank, normalized to [0, 1]
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / (n - 1)
    
    # Define the top 5% threshold
    tau = 0.95
    
    # Normalized position within the top-5% band: 0 at tau, 1 at 1.0
    # For points below tau, this will be negative; we'll clip to 0
    band_pos = np.clip((sigma_ranks - tau) / (1.0 - tau), 0.0, 1.0)
    
    # Valid candidates: in top 5% uncertainty AND mean below current best
    in_top_band = sigma_ranks >= tau
    below_best = mu < f_best
    valid = in_top_band & below_best
    
    # Alpha controls the exploration bonus relative to the mean improvement
    # We scale alpha by the typical sigma magnitude to make it adaptive
    sigma_median = np.median(sigma)
    alpha = 2.0 * sigma_median
    
    # Build scores
    # For valid candidates: -mu + alpha * band_pos
    # For invalid candidates: a low finite score based on how close they are to being valid
    scores = np.empty(n, dtype=np.float64)
    
    # Invalid: give a score that decreases as we move away from the valid region
    # Use -mu - penalty for those not in top band or not below best
    scores[:] = -mu - 5.0 * sigma_median  # baseline low score
    
    # For valid candidates, use the proper acquisition score
    scores[valid] = -mu[valid] + alpha * band_pos[valid]
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
