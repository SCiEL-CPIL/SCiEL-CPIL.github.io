import numpy as np
import time

# Global start time for the deadline guard
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # Seconds

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses a resource fragmentation heuristic:
    Score = -harmonic_mean(remaining, item)
    This penalizes both tiny leftover gaps (waste) and excessively large remaining spaces.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Return a default "safe" score, e.g., Best Fit
        remaining = bins - item
        valid_mask = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores

    # Track remaining capacity for each bin
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    if n_valid == 1:
        scores[valid_mask] = 0.0
        return scores
    
    # Resource fragmentation heuristic:
    # harmonic_mean(a, b) = 2*a*b / (a+b)
    # We want to minimize the harmonic mean of remaining and item size.
    # This is low when one of the values is small.
    # If remaining is small (tight fit), harmonic mean is small.
    # If item is small relative to remaining, harmonic mean is small.
    # Wait, if remaining is large and item is small, harmonic mean is ~2*item.
    # If remaining is small (tight), harmonic mean is ~2*remaining.
    # So minimizing harmonic mean favors tight fits OR small items in large bins?
    # Actually, harmonic mean is dominated by the smaller of the two.
    # So if item << remaining, H ~ 2*item.
    # If remaining << item (impossible since remaining >= 0 and item fits, so remaining >= 0),
    # actually remaining can be 0.
    # If remaining is close to 0, H is close to 0.
    # If remaining is large, H is close to 2*item.
    # So minimizing H favors bins where remaining is small (tight fit).
    # This is similar to Best Fit.
    # But the directive says: "penalizes both tiny leftover gaps (waste) and excessively large remaining spaces (risk of future fragmentation)"
    # Minimizing H does NOT penalize tiny leftover gaps; it FAVORS them (H is small).
    # Let's re-read the directive carefully.
    # "scoring bins with the negative of the harmonic mean of their remaining capacity and the incoming item size, which penalizes both tiny leftover gaps (waste) and excessively large remaining spaces"
    # This implies the score should be LOW for tiny gaps and LOW for large gaps.
    # So the score should be HIGH for intermediate gaps?
    # Or maybe I misinterpreted "negative of the harmonic mean".
    # If Score = -H, then:
    # Tiny gap (remaining ~ 0) -> H ~ 0 -> Score ~ 0.
    # Large gap (remaining >> item) -> H ~ 2*item -> Score ~ -2*item.
    # So Score is higher for tiny gaps. This favors Best Fit.
    # Does it penalize tiny gaps? No, it rewards them.
    # Does it penalize large gaps? Yes, Score is more negative.
    # So this heuristic is essentially Best Fit with a penalty for large bins.
    # But Best Fit is already the baseline (score 1.0).
    # Maybe the harmonic mean behaves differently?
    # Let's try a different interpretation.
    # Perhaps the "resource fragmentation" idea is to avoid creating small fragments.
    # If we place an item in a bin, the remaining space is the fragment.
    # We want to avoid small fragments because they can't hold future items.
    # So we want remaining space to be either 0 (perfect fit) or large enough to hold future items.
    # A score that is high for remaining=0 and high for remaining=large, and low for remaining=small?
    # That would be a U-shaped curve.
    # But the directive specifically says "negative of the harmonic mean".
    # Let's stick to the directive's formula: Score = -H(remaining, item).
    # And see if it improves.
    
    # Calculate harmonic mean
    # H = 2 * a * b / (a + b)
    # To avoid division by zero, we can add a small epsilon.
    # But if remaining is 0, H is 0.
    # If item is 0, H is 0.
    # Since item > 0, we only care about remaining.
    
    a = valid_remaining
    b = np.full_like(a, item)
    
    # H = 2 * a * b / (a + b)
    # If a + b is 0, H is 0.
    # Since item > 0, a + b > 0 always.
    
    H = 2.0 * a * b / (a + b)
    
    scores[valid_mask] = -H
    
    return scores
