"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Next-Item Awareness heuristic:
    Estimates the expected size of the next item from the current bin occupancy
    distribution, then scores bins by how well their post-placement remaining
    space matches that predicted size. Bins whose remaining space is close to
    the predicted next item size get higher scores (less waste expected).
    """
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # 1. Identify valid bins (those with enough capacity for the current item)
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
    
    valid_bins = bins[valid_mask]
    new_rems = valid_bins - item
    
    # 2. Estimate the expected size of the next item from bin occupancy
    # Use the mean of remaining capacities in partially-filled bins as a proxy
    # for typical item sizes, since gaps tend to be filled by similar-sized items.
    # Also consider the current item size as a signal.
    used_mask = (bins > 0) & (bins < 100)
    if np.any(used_mask):
        used_rems = bins[used_mask]
        # Weighted estimate: blend median remaining capacity with current item size
        med_rem = np.median(used_rems)
        mean_rem = np.mean(used_rems)
        # The next item is likely to be around the typical gap size,
        # but also influenced by what's currently arriving
        s_next = 0.5 * med_rem + 0.3 * mean_rem + 0.2 * item
    else:
        # No partially filled bins; assume next item is similar to current
        s_next = item
    
    # Clamp to reasonable range
    s_next = np.clip(s_next, 1.0, 99.0)
    
    # 3. Score each valid bin by how well its remaining space matches s_next
    # We want new_rem to be close to s_next (so the next item fits well)
    # OR very small (tight fit, minimal waste)
    # Use a quadratic penalty for deviation from ideal, with a bonus for tight fits
    
    # Ideal: new_rem ≈ s_next (next item fits with little waste)
    # Good: new_rem ≈ 0 (tight fit, no room for anything but minimal waste)
    # Bad: 0 < new_rem < s_next (gap too small for next item, wasted space)
    # Bad: new_rem >> s_next (large unused space, potential for future waste)
    
    # Compute match score: how close is new_rem to s_next?
    # Use exponential decay for a smooth preference
    match_score = -np.abs(new_rems - s_next)
    
    # Bonus for very tight fits (new_rem close to 0)
    tight_bonus = -new_rems * 0.5
    
    # Penalty for dead-zone gaps (0 < new_rem < s_next)
    dead_zone = (new_rems > 0) & (new_rems < s_next)
    dead_penalty = np.where(dead_zone, -20.0 * (s_next - new_rems) / max(s_next, 0.1), 0.0)
    
    # Combine scores
    valid_scores = match_score + tight_bonus + dead_penalty
    
    # Assign scores back to the output array
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
