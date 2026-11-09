import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_after = bins - item
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    valid_mask = remaining_after >= 0
    if not np.any(valid_mask):
        return scores
        
    r_new = remaining_after[valid_mask]
    
    # Base score: Best-Fit (prefer smaller remaining capacity)
    base_scores = -r_new
    
    # Fragmentation-aware penalty based on ratio of item to remaining capacity
    # We penalize bins where the item occupies a moderate fraction (20-50%)
    # of the remaining space, as this creates small, potentially unusable fragments
    penalty = np.zeros_like(r_new)
    
    # Calculate ratio: item / (item + remaining_after) = item / original_capacity_used
    # Actually, we want: item / (bins[valid] - item + item) = item / bins[valid]
    # But simpler: ratio = item / (item + r_new)
    # When item is 20-50% of the bin's used portion, we're in the danger zone
    # Let's use: ratio = item / (item + r_new)
    # If ratio is between 0.2 and 0.5, we create a fragment that's 50-80% of original
    # Actually, let's think differently:
    # After placing item, remaining = r_new
    # The "fragment" is r_new. We want to avoid r_new being in a range where it's 
    # too small to be useful but wastes space.
    # 
    # Key insight: penalize when item is a moderate fraction of the bin capacity
    # Let bin_cap = bins[valid_mask]
    # ratio = item / bin_cap
    # If ratio is in [0.2, 0.5], the remaining is in [0.5, 0.8] * bin_cap
    # For Weibull with capacity 100, this means remaining in [50, 80] which is still useful
    # 
    # Better approach: penalize based on whether the resulting fragment is "wasteful"
    # A fragment is wasteful if it's small enough that few future items can fit
    # For Weibull(100), items are typically 10-50, so fragments < 15 are often unusable
    # 
    # Let's use: penalty peaks when r_new is in [15, 35] (sweet spot for waste)
    # This is similar to the original but let's refine the parameters
    
    # Actually, following the directive more closely:
    # Use ratio = item / bins[valid_mask] (fraction of bin the item takes)
    bin_caps = bins[valid_mask]
    ratio = item / bin_caps
    
    # Penalize when ratio is in [0.2, 0.5] - item takes moderate fraction
    # This leaves a fragment that's 50-80% of bin, which might be too large to be 
    # efficiently used if future items are small
    # 
    # Wait, let me reconsider. The goal is to avoid creating SMALL unusable fragments.
    # Small fragments occur when r_new is small (e.g., < 20).
    # r_new is small when item is large relative to bin capacity.
    # So we should penalize when item/bins is HIGH (e.g., > 0.7), not moderate.
    # 
    # Let me re-read the directive: "targeting bins where the item is a moderate fraction 
    # (e.g., 20-50%) of the remaining space"
    # 
    # "remaining space" = bins[valid_mask] (before placing item)
    # So ratio = item / bins[valid_mask]
    # If ratio is 20-50%, then after placing, remaining = bins * (1 - ratio) = 50-80% of bin
    # For bin=100, remaining = 50-80. These are LARGE fragments, not small.
    # 
    # I think the directive means: we want to AVOID creating fragments that are 
    # in a "bad" range. Let me interpret it as:
    # Penalize when the resulting fragment (r_new) would be in a range where it's 
    # not quite small enough to be negligible but not large enough to be very useful.
    # 
    # For Weibull(100), let's say fragments in [10, 30] are problematic.
    # This happens when item is in [70, 90] for a 100-capacity bin.
    # 
    # Let's use a smooth penalty function that peaks at r_new = 20
    
    # Penalty function: peaks at r_new = 20, zero at r_new = 0 and r_new = 40
    center = 20.0
    half_width = 20.0  # penalty is zero at center ± half_width
    
    dist_from_center = np.abs(r_new - center)
    in_penalty_zone = dist_from_center < half_width
    
    if np.any(in_penalty_zone):
        # Smooth bell-shaped penalty
        x = dist_from_center[in_penalty_zone] / half_width
        p_val = 30.0 * (1 - x**2)  # Parabolic, max at center
        penalty[in_penalty_zone] = p_val
    
    # Additional penalty: if r_new is very small (< 5), it's almost wasted
    tiny_fragment = r_new < 5.0
    if np.any(tiny_fragment):
        penalty[tiny_fragment] += 10.0 * (1 - r_new[tiny_fragment] / 5.0)
    
    final_scores = base_scores - penalty
    scores[valid_mask] = final_scores
    
    return scores
