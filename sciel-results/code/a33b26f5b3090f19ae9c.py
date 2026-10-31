"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

# Capacity constant
_CAPACITY = 100.0
_HALF_CAP = _CAPACITY / 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    
    Strategy: Polarization with Sliver Avoidance.
    We prefer bins that, after placing the item, leave a remainder that is
    either very small (bin nearly full) or very large (bin mostly empty).
    This avoids creating "half-full" bins which are hard to fill efficiently.
    
    However, we must avoid creating tiny "slivers" (e.g., < 5 units) that are
    likely unusable for future items. So we penalize remainders that are small
    but not small enough to be considered "full" (i.e., below a threshold).
    
    The score is a combination of:
    1. Polarization: Distance from half-capacity (squared).
    2. Sliver Penalty: Strong penalty for remainders in (0, sliver_threshold).
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # --- Polarization Score ---
    # We want remainders close to 0 or close to Capacity.
    # Distance from half-capacity, squared.
    dist_from_half = valid_remaining - _HALF_CAP
    pol_score = dist_from_half ** 2
    
    # --- Sliver Avoidance Penalty ---
    # If the remaining space is very small (a sliver), it's likely wasted.
    # We define a sliver as remaining < 5.0 (5% of capacity).
    # We apply a penalty that increases as the remainder gets closer to 0,
    # but only for remainders below the threshold.
    sliver_threshold = 5.0
    
    penalty = np.zeros_like(valid_remaining)
    tight_mask = valid_remaining < sliver_threshold
    
    if np.any(tight_mask):
        # For remainders below threshold, apply a quadratic penalty.
        # The penalty is 0 at the threshold and increases as remainder -> 0.
        # This ensures that a bin with 4.9 remaining is slightly worse than
        # a bin with 5.1 remaining (which gets no penalty), but a bin with
        # 0.1 remaining is much worse.
        # 
        # Let's use a penalty that is proportional to (threshold - remainder)^2
        # scaled by a factor to make it significant compared to pol_score.
        # pol_score range: [0, 2500] (for remainder 0 or 100, dist=50, sq=2500)
        # We want the penalty to be able to override the polarization preference
        # for tiny slivers. 
        # If remainder is 1, pol_score = (1-50)^2 = 2401.
        # If we add a penalty of, say, 3000, the total score becomes -599.
        # A bin with remainder 5.1 has pol_score = (5.1-50)^2 = 2016, penalty 0. Score 2016.
        # So 5.1 is preferred over 1. Good.
        
        diff = sliver_threshold - valid_remaining[tight_mask]
        # Scale factor: 1000 seems reasonable to make slivers very unattractive.
        penalty[tight_mask] = 1000.0 * (diff ** 2)
        
    # Final score: Polarization - Penalty
    final_scores = pol_score - penalty
    
    # Assign scores back to the original array
    scores[valid_mask] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
