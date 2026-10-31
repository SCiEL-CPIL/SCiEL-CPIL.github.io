import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Structural change: Combines 'gap polarization' (from inspirations) with
    'probabilistic fragmentation penalty'. 
    
    Core Idea:
    1. Polarization: Prefer bins that leave a remainder either very small (tight fit,
       maximizing immediate utilization) or very large (empty bin, preserving for
       future large items). This is modeled by the squared distance from the
       midpoint of the *valid* remaining capacities (or a fixed capacity center).
       Using the global capacity (100) center is robust.
       
    2. Fragmentation Penalty: Weibull items have a heavy tail of small items but
       a significant probability of medium items. Creating a "dead" gap that is
       too small to fit the *expected* next few items is bad. 
       We estimate the expected item size (mean of Weibull) and penalize gaps
       that are smaller than this expected size but larger than a "sliver" 
       threshold. This encourages either filling the bin completely (gap ~ 0)
       or leaving a gap large enough to be useful (gap > expected_item).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        # Fallback: simple best fit
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid):
        return scores

    valid_remaining = remaining[valid]
    valid_indices = np.where(valid)[0]

    # --- Structural Component 1: Polarization ---
    # We want to push remainders towards 0 or Capacity (100).
    # The "center" of the capacity range is 50.
    # Score is higher when remainder is far from 50.
    # Using squared distance gives a stronger preference for extremes.
    capacity = 100.0
    center = capacity / 2.0
    
    # Distance from center
    dist_from_center = valid_remaining - center
    polarization_score = dist_from_center ** 2
    
    # --- Structural Component 2: Probabilistic Fragmentation Penalty ---
    # Weibull distribution with shape k~1.5-2.0 and scale lambda~10-15 typically.
    # Mean = lambda * Gamma(1 + 1/k).
    # Let's assume a generic "expected useful gap" threshold.
    # If a gap is < ~10-15 units, it's likely too small for a significant fraction
    # of future items, creating fragmentation.
    # However, if a gap is < ~2 units, it's effectively "filled" and acceptable.
    
    # We define a "useless zone" between a small sliver threshold and a useful gap threshold.
    sliver_threshold = 2.0
    useful_gap_threshold = 12.0  # Approximate mean item size for typical Weibull params
    
    # Penalty logic:
    # If gap < sliver_threshold: No penalty (it's a tight fit, good).
    # If sliver_threshold <= gap <= useful_gap_threshold: Linear penalty, max at useful_gap_threshold?
    # Actually, we want to penalize gaps that are "stuck" in the middle of small sizes.
    # Let's penalize gaps in [sliver_threshold, useful_gap_threshold] most heavily.
    # Gaps > useful_gap_threshold are fine (large enough for future items).
    
    penalty = np.zeros_like(valid_remaining)
    
    # Mask for gaps in the "danger zone" (too big for sliver, too small for useful)
    danger_mask = (valid_remaining > sliver_threshold) & (valid_remaining < useful_gap_threshold)
    
    if np.any(danger_mask):
        # Create a penalty that peaks in the middle of this range or is linear?
        # A quadratic penalty that is 0 at boundaries and max in middle might be complex.
        # Simpler: Linear penalty that increases as we move away from 0, but resets?
        # Let's try: Penalize based on how "stuck" it is.
        # A gap of 5 is worse than a gap of 2 (sliver) or 15 (useful).
        # Let's use a triangular penalty shape:
        # 0 at sliver_threshold, 0 at useful_gap_threshold, Max in middle.
        mid_point = (sliver_threshold + useful_gap_threshold) / 2.0
        width = (useful_gap_threshold - sliver_threshold) / 2.0
        
        # Distance from mid-point, normalized to [0, 1]
        # 1 at mid-point, 0 at boundaries
        dist_to_mid = np.abs(valid_remaining[danger_mask] - mid_point)
        norm_dist = 1.0 - (dist_to_mid / width)
        
        # Penalty magnitude: Strong enough to override polarization for these gaps.
        # Polarization score for gap=7 (mid) is (7-50)^2 = 1849.
        # We need a penalty > 1849 to flip the decision if this gap is worse than a sliver.
        # A sliver (gap=1) has polarization (1-50)^2 = 2401.
        # So a penalty of ~600-1000 should make gap=7 less attractive than gap=1.
        penalty_scale = 1500.0
        penalty[danger_mask] = norm_dist * penalty_scale

    # --- Combine Scores ---
    # Higher score is better.
    # Polarization favors 0 and 100.
    # Penalty hurts "dead" gaps.
    
    final_scores = polarization_score - penalty
    
    scores[valid_indices] = final_scores
    
    return scores
