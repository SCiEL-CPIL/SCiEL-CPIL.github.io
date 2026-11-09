"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 115.0  # 5s safety margin
    
    remaining = bins - item
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    n_valid = len(valid_remaining)
    
    # Fragmentation-aware scoring:
    # The key insight is that leaving a "dead zone" remainder is bad.
    # A dead zone is a remainder that is too small to fit most future items
    # but too large to be completely wasted. For Weibull items with capacity 100,
    # items tend to be small-to-medium. A remainder of ~5-30 is often unusable.
    #
    # Strategy: Score = -remaining (prefer tight fits) + fragmentation penalty
    # The fragmentation penalty peaks when remaining/item is in a "bad ratio" range
    # where the leftover space is likely unusable.
    
    if n_valid == 1:
        scores[valid] = -valid_remaining
        return scores
    
    # Base score: prefer tighter fits (less remaining = better)
    # Use negative remaining as primary signal
    base_score = -valid_remaining
    
    # Fragmentation penalty: penalize remainders that create "dead zones"
    # For Weibull-distributed items with capacity 100, typical items are ~5-50
    # A remainder that is 10-40 units is often a dead zone (too small for many items)
    # We model this as a penalty that peaks in the dead-zone range
    
    # Ratio of item to original bin remaining (before placing this item)
    # item / (bins[valid]) tells us how much of the available space we're using
    # If item is a small fraction, the remainder is large (good, flexible)
    # If item is a large fraction, remainder is small (tight fit, good)
    # Bad case: item is moderate fraction, leaving a moderate remainder (dead zone)
    
    # Estimate original remaining capacity for valid bins
    # bins[valid] is the remaining capacity BEFORE placing the item
    orig_remaining = bins[valid]
    
    # Ratio: what fraction of available space does this item consume?
    # Guard against division by zero
    safe_orig = np.maximum(orig_remaining, 1e-9)
    ratio = item / safe_orig  # 0 to 1 (item fits, so <= 1)
    
    # Fragmentation penalty based on the leftover space characteristics
    # The leftover = valid_remaining = orig_remaining - item
    # We want to penalize leftovers that are in a "dead zone"
    
    # Model: dead zone occurs when leftover is between ~15% and ~50% of typical item sizes
    # For Weibull with capacity 100, let's say typical items are around 10-40
    # So dead zone leftover: roughly 10-40 units
    
    # Use a smooth penalty function:
    # Penalty peaks when leftover is in [10, 40] range
    # We'll use a Gaussian-like bump centered at 25 with width ~15
    
    leftover = valid_remaining
    dead_zone_center = 25.0
    dead_zone_width = 15.0
    
    # Smooth bump: exp(-(leftover - center)^2 / (2*width^2))
    # This peaks at center and decays away
    bump = np.exp(-((leftover - dead_zone_center) ** 2) / (2 * dead_zone_width ** 2))
    
    # Scale the penalty: max penalty of ~1.0 units (comparable to base score scale)
    fragmentation_penalty = 1.0 * bump
    
    # Also consider: if the item is a very small fraction of available space
    # (ratio < 0.2), the remainder is large and flexible - no penalty
    # If item is a very large fraction (ratio > 0.8), tight fit - no penalty
    # Moderate ratios (0.3-0.7) with moderate remainders are the problematic case
    
    # Additional ratio-based modulation of the penalty
    # Reduce penalty when ratio is very low (large remainder, flexible) or very high (tight fit)
    ratio_mod = 1.0 - np.exp(-((ratio - 0.5) ** 2) / (2 * 0.15 ** 2))
    # ratio_mod is 0 when ratio=0.5 (moderate, most problematic), 1 when ratio is far from 0.5
    # We want to REDUCE penalty when ratio is far from 0.5, so multiply by (1 - ratio_mod)
    # Actually: penalty should be HIGH when ratio is moderate AND leftover is in dead zone
    # So: final_penalty = fragmentation_penalty * (1 - ratio_mod)
    # When ratio=0.5: ratio_mod=0, penalty = fragmentation_penalty (full)
    # When ratio=0.1 or 0.9: ratio_mod≈1, penalty ≈ 0 (reduced)
    
    final_penalty = fragmentation_penalty * (1.0 - ratio_mod)
    
    # Final score: base (prefer tight) minus fragmentation penalty
    # Higher score = better
    scores_valid = base_score - final_penalty
    
    scores[valid] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
