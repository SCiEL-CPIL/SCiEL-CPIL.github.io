"""Bootstrap portfolio: SEARCH approach with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    The evaluator places the item in the bin with the HIGHEST score.
    We want to minimize mean percent excess over the L1 lower bound.
    
    Strategy: Use a sophisticated heuristic that balances:
    1. Best-fit behavior (minimize wasted space) when bins are tight
    2. Avoid creating too many small fragments
    3. Consider the global state of bin utilization
    """
    start = time.time()
    DEADLINE = start + 115.0  # safety margin
    
    eps = 1e-9
    capacity = 100.0
    
    # Get remaining capacities
    remaining = bins.copy()
    valid_mask = remaining >= item - eps
    
    # If no valid bin, return all inf (shouldn't happen in practice)
    if not np.any(valid_mask):
        return np.full_like(bins, np.inf)
    
    # Compute various features for each valid bin
    # We want to return scores where HIGHER is better (evaluator picks max)
    
    # Feature 1: Remaining space after placement (lower is better for best-fit)
    # So we negate it: higher score = less remaining = tighter fit
    remaining_after = remaining - item
    
    # Feature 2: How "full" the bin is before placement (higher = more full = better)
    fill_before = 1.0 - remaining / capacity
    
    # Feature 3: Penalize creating very small fragments
    # If remaining_after is very small, that's good (tight fit)
    # If remaining_after is large, that's bad (wastes space)
    # But we also don't want to leave bins with just a tiny bit of space
    # that can't fit any future items
    
    # Compute a composite score
    # We want to prefer bins that:
    # - Have tight fit (small remaining_after) - this is best-fit
    # - But not TOO tight if it creates unusable fragments
    
    # Approach: Use a weighted combination
    # Score = w1 * (-remaining_after) + w2 * fill_before + w3 * penalty_for_fragmentation
    
    # Fragmentation penalty: penalize if remaining_after is in a "dead zone"
    # where it's too small to be useful but too large to be a tight fit
    # For Weibull items with capacity 100, typical items are ~10-30
    # So remaining_after < 10 might be too small to fit most future items
    # remaining_after > 50 is clearly wasteful
    
    # Let's use a smooth penalty
    # Ideal remaining_after: small but not zero
    # We can model this as: score is high when remaining_after is close to 0
    # but drops off if it's in the "useless" range
    
    # Simple approach: Best-fit with a slight preference for bins that are
    # already partially filled (to avoid opening new bins unnecessarily)
    
    # Primary: best-fit (minimize remaining_after)
    # Secondary: prefer bins with higher fill_before (to consolidate)
    
    # Normalize remaining_after to [0, 1]
    max_remaining = np.max(remaining_after[valid_mask]) if np.any(valid_mask) else 1.0
    if max_remaining < eps:
        max_remaining = 1.0
    
    normalized_remaining = remaining_after / max_remaining
    
    # Score: higher is better
    # Base score: negative of normalized remaining (best-fit)
    base_score = -normalized_remaining
    
    # Bonus for bins that are already fairly full
    fill_bonus = 0.3 * fill_before
    
    # Penalty for creating unusable fragments (remaining_after in [5, 20])
    # This is a heuristic: if you leave 5-20 space, it might not fit many future items
    # We use a Gaussian-like penalty centered at 12.5 (middle of 5-20)
    fragment_center = 12.5
    fragment_width = 8.0
    fragment_penalty = -0.2 * np.exp(-((remaining_after - fragment_center) ** 2) / (2 * fragment_width ** 2))
    
    # Combine
    scores = base_score + fill_bonus + fragment_penalty
    
    # Mask out invalid bins
    scores = np.where(valid_mask, scores, np.inf)
    
    # Ensure we return a valid array
    return scores
# EVOLVE-BLOCK-END
