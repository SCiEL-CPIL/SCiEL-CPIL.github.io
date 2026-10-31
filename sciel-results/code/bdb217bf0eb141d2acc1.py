import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    rem = bins[feasible] - item
    
    # Smooth adaptive scoring:
    # Base: quadratic best fit (strongly prefer tight fits)
    # Adjustment: for very small remainders (< ~5% of capacity), the bin is
    # effectively "sealed" — any small remainder is similarly bad. We soften
    # the penalty using a smooth transition so that remainders near 0 don't
    # get disproportionately penalized compared to slightly larger ones.
    #
    # Mechanism: 
    #   - For large remainders, quadratic penalty dominates (tight packing).
    #   - For very small remainders, we blend toward a linear penalty to avoid
    #     over-penalizing the "last bit" of space that can't be used anyway.
    #   - The transition is smooth via a sigmoid centered at ~5% of capacity.
    
    # Threshold for "effectively sealed" bin (as fraction of capacity)
    threshold = 0.05 * C  # 5 units
    
    # Smooth weight: 1.0 for large rem, 0.5 for small rem (blends quad toward linear)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # We want: when rem >> threshold, weight ≈ 1 (full quadratic)
    #          when rem << threshold, weight ≈ 0.5 (half quadratic, half linear)
    x = (rem - threshold) / (0.02 * C)  # scale for smoothness
    weight = 0.5 + 0.5 * (1.0 / (1.0 + np.exp(-x)))
    
    # Quadratic component: -rem^2
    # Linear component: -rem
    # Blended score: -weight * rem^2 - (1 - weight) * rem
    # But we want higher score = better, and we want to prefer SMALL rem.
    # So score should be NEGATIVE and more negative for larger rem.
    # Let's use: score = -(weight * rem^2 + (1 - weight) * rem)
    
    quad_part = weight * rem * rem
    lin_part = (1.0 - weight) * rem
    scores[feasible] = -(quad_part + lin_part)
    
    return scores
