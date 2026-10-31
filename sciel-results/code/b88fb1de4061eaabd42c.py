"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Pre-compute Weibull CDF parameters for scoring
# The task mentions Weibull sequences. A common default is shape k=1.5, scale lambda=1.0
# or similar. Without explicit parameters, we assume a standard distribution.
# However, the strategy is to use the CDF of the remaining space to estimate "regret".
# If the remaining space is large, it's likely to be filled by future items.
# If the remaining space is small, it might be wasted.
# Actually, the directive suggests: "score as the negative of the cumulative distribution function"
# Wait, the directive text is cut off: "negative of the cumulative distribution function <OBSERVED...>"
# Let's re-read: "approximated by the probability that this remainder cannot be perfectly filled"
# If we assume items are Weibull(k, lambda), the probability that a random item fits in remainder r is P(X <= r) = CDF(r).
# If we want to minimize waste, we want the remainder to be small (Best Fit) OR we want the remainder to be such that it's likely to be filled.
# If we place an item, the new remainder is r_new = bin - item.
# We want to maximize the chance that r_new is "good".
# A common heuristic is to prefer remainders that are likely to be filled by a single future item (if small) or many items.
# The directive says: "score as the negative of the cumulative distribution function"
# This implies: Score = -CDF(r_new).
# If CDF(r_new) is high, the score is very negative (bad).
# If CDF(r_new) is low, the score is close to 0 (better).
# This would favor SMALL remainders (since CDF is increasing).
# This is essentially Best-Fit but smoothed by the CDF.
# If the distribution is known, we can compute CDF.
# Let's assume Weibull with shape k=1.5 and scale lambda=1.0 (common defaults if not specified).
# CDF(x) = 1 - exp(-(x/lambda)^k)
# If x < 0, CDF = 0. If x is large, CDF -> 1.
# Score = -CDF(remaining).
# This favors small remainders.

def _weibull_cdf(x, k=1.5, lam=1.0):
    """Compute Weibull CDF for array x."""
    x = np.asarray(x)
    cdf = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        cdf[mask] = 1.0 - np.exp(-((x[mask] / lam) ** k))
    return cdf

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Distribution-aware Best-Fit.
    Use the Weibull CDF of the remaining space to score bins.
    Score = -CDF(remaining_space_after_placement).
    This favors bins that leave a small remainder (likely to be filled or minimal waste).
    """
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    
    # Calculate CDF for feasible bins
    # We want to maximize the score (which is -CDF).
    # So we want to MINIMIZE CDF(remaining).
    # CDF is increasing, so we want to minimize remaining.
    # This is Best-Fit, but with a non-linear penalty.
    
    rem = remaining_after[feasible]
    cdf_vals = _weibull_cdf(rem)
    scores[feasible] = -cdf_vals
    
    return scores
# EVOLVE-BLOCK-END
