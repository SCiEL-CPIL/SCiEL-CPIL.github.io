"""Improved bin packing strategy using polarization scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores (higher is better for selection).
    
    Strategy: Polarization scoring
    - Maximizes distance from half-capacity (50).
    - Small items: prefer bins that are already nearly full (best-fit behavior)
      to consolidate small items and avoid fragmentation.
    - Large items: prefer bins that are nearly empty (worst-fit behavior)
      to preserve half-full bins for future medium items.
    - This adapts between Best-Fit and Worst-Fit based on item size.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple best-fit
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(mask):
            scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0

    scores = np.full(len(bins), -np.inf)
    if not np.any(mask):
        return scores

    # Only score bins that can fit the item
    valid_remaining = remaining[mask]
    valid_bins = bins[mask]
    
    # Polarization: maximize distance from 50 (half of capacity 100)
    # For small items (item < 10): prefer bins with small remaining (best-fit)
    #   -> bins with remaining close to 0 are "polarized" towards full
    # For large items (item > 90): prefer bins with large remaining (worst-fit)
    #   -> bins with remaining close to 100 are "polarized" towards empty
    # For medium items: balance
    
    # The key insight: we want to maximize |remaining - 50|
    # But we also need to ensure the item fits, so remaining >= item
    
    # Actually, let's think about this more carefully:
    # If item is small (say 5), and we have bins with remaining [3, 50, 95]:
    #   - remaining=3: item doesn't fit (3 < 5), so mask is false
    #   - remaining=50: fits, |50-50|=0
    #   - remaining=95: fits, |95-50|=45
    #   We'd prefer remaining=95 (worst-fit), which preserves the half-full bin.
    
    # If item is large (say 95), and we have bins with remaining [5, 50, 95]:
    #   - remaining=5: doesn't fit
    #   - remaining=50: doesn't fit
    #   - remaining=95: fits, |95-50|=45
    #   Only one option.
    
    # If item is medium (say 50), and we have bins with remaining [50, 90]:
    #   - remaining=50: fits, |50-50|=0
    #   - remaining=90: fits, |90-50|=40
    #   We'd prefer remaining=90.
    
    # The polarization score is: -|remaining - 50|
    # This encourages bins to be either nearly full or nearly empty,
    # avoiding the "half-full" state which is wasteful.
    
    polarization = -np.abs(valid_remaining - 50.0)
    
    # Add a small tie-breaker: among equally polarized bins, prefer the one
    # that minimizes remaining (to pack tightly when polarization is similar)
    # This helps when multiple bins have similar distance from 50.
    tie_breaker = -valid_remaining * 0.01
    
    valid_scores = polarization + tie_breaker
    scores[mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
