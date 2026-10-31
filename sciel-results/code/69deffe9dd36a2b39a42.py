"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # New Strategy: Golden Ratio Attractor
    # Instead of linear Best Fit or dynamic phase-based weighting, we model the
    # bin state as a 1D spatial coordinate. We aim to place items such that the
    # resulting remainder (gap) is close to a "golden ratio" target relative to
    # the capacity.
    
    # The golden ratio (phi ≈ 0.618) is often cited in subdivision problems as
    # minimizing worst-case waste or providing a stable attractor that prevents
    # fragmentation into too-small gaps while not leaving too-large gaps that
    # waste space.
    
    # We assume a fixed capacity of 100 (as per task description).
    # Target remainder = 0.618 * 100 = 61.8? 
    # Wait, if we put an item in, the remainder is `bins - item`.
    # If we want the remainder to be near 61.8, that means we prefer bins where
    # `bins - item` is close to 61.8.
    # This implies `bins` should be close to `item + 61.8`.
    
    # Let's re-evaluate. 
    # If we use Best Fit, we minimize `bins - item`.
    # If we use Worst Fit, we maximize `bins - item`.
    # If we use a "Golden Fit", we minimize `|bins - item - target_gap|`.
    
    # What is the optimal target gap?
    # For Weibull distributions, items are often smaller than the mean.
    # If we leave a gap of ~61.8, that's a lot of space. 
    # Maybe the target should be smaller?
    # The prompt suggests: "score is the negative Euclidean distance between the resulting remainder and a 'golden ratio' target (approx. 0.618 * capacity)".
    
    # Let's follow the prompt's suggestion exactly for this structural change.
    # Capacity C = 100.
    # Target Gap G = 0.618 * C = 61.8.
    
    # Score = -| (bins - item) - G |
    
    # This will prefer bins where the remaining space after placing the item is
    # close to 61.8.
    
    # Let's check the implications:
    # If item is small (e.g., 5), we prefer bins with ~66.8 remaining.
    # If item is large (e.g., 50), we prefer bins with ~111.8 remaining (impossible if C=100).
    # So for large items, it will just pick the largest valid bins (Worst Fit behavior).
    # For small items, it will pick bins that are moderately filled, avoiding
    # both very full bins (small gap) and very empty bins (large gap).
    
    # This is a distinct structural change from linear scoring.
    
    C = 100.0
    G = 0.618 * C
    
    # Calculate the resulting remainder for valid bins
    valid_remaining = bins[valid]
    gaps = valid_remaining - item
    
    # Calculate score: negative distance to target gap
    # We use Euclidean distance (absolute difference)
    scores = -np.abs(gaps - G)
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END
