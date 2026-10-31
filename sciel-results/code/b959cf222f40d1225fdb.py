"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timing state
_START_TIME = time.time()
_DEADLINE = 115.0  # 120s limit, 5s safety margin

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Adaptive bin selection combining Best-Fit with Weibull-aware fragmentation penalty.
    
    Core ideas:
      1. Primary signal: Best-Fit (minimize leftover after placing the item).
      2. Weibull CDF-aware penalty: Penalize remainders that are unlikely to be 
         filled by future items based on the Weibull distribution.
      3. Smooth transition between "tight fit" and "wasted gap" regimes.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Safety fallback: simple Best Fit
        remaining = bins - item
        scores = np.full_like(bins, -np.inf)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    capacity = 100.0

    # Compute remaining capacity if we place the item in each bin
    remaining = bins - item

    # Valid bins are those with remaining >= 0
    valid_mask = remaining >= 0

    # If no valid bin, return -inf for all
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)

    # Initialize scores
    scores = np.full_like(bins, -np.inf)

    # Get the current fill levels of valid bins
    valid_bins = bins[valid_mask]
    valid_remaining = remaining[valid_mask]

    # Estimate a "useful gap" threshold from the data.
    # For Weibull items with typical shape k~1.5-2.5, scale ~10-20,
    # mean item sizes are ~8-15.
    if len(valid_bins) > 0:
        # Use a robust estimate of typical small-item size
        q25 = np.percentile(valid_bins, 25)
        small_item_est = np.clip(q25, 3.0, 15.0)
    else:
        small_item_est = 10.0

    # Weibull CDF for estimating probability that a remainder can be filled
    # Weibull CDF: F(x) = 1 - exp(-(x/lambda)^k)
    # For k=1.5, lambda=small_item_est (scale parameter)
    k_weibull = 1.5
    lambda_weibull = small_item_est

    # Compute Weibull CDF for valid remainders
    # This gives the probability that a random future item fits in the remainder
    # High CDF means high chance of being filled -> good
    # Low CDF means low chance of being filled -> bad (wasted space)
    rem_safe = np.maximum(valid_remaining, 1e-10)
    weibull_cdf = 1.0 - np.exp(-((rem_safe / lambda_weibull) ** k_weibull))

    # "Dead zone" threshold: gaps smaller than this are likely wasted
    # but not so small that they're effectively zero
    dead_zone = 0.5 * small_item_est

    # Smooth penalty for remainders in the "dead zone"
    # Use a Gaussian-shaped penalty centered in the dead zone
    sigma = 0.3 * small_item_est
    gap_center = dead_zone * 0.5  # center of the dead zone
    
    gap_deviation = valid_remaining - gap_center
    gaussian_penalty = np.exp(-0.5 * (gap_deviation / sigma) ** 2)

    # Scale the penalty based on fragmentation risk
    n_valid = len(valid_bins)
    n_total = len(bins)
    frag_weight = 1.0 * (1.0 - np.exp(-0.1 * n_valid)) if n_total > 0 else 0.5

    # Base score: Best-Fit (negative of remaining, so tighter = higher score)
    base_score = -1.0 * valid_remaining

    # Weibull-aware bonus: reward remainders with high probability of being filled
    # This encourages using bins where the leftover is likely to be useful
    weibull_bonus = 0.5 * weibull_cdf

    # Consolidation bonus for fuller bins
    consolidation = 0.02 * valid_bins

    # Combine all components
    frag_penalty = frag_weight * gaussian_penalty
    scores_valid = base_score + weibull_bonus + consolidation - frag_penalty

    # Assign scores
    scores[valid_mask] = scores_valid

    return scores

# EVOLVE-BLOCK-END
