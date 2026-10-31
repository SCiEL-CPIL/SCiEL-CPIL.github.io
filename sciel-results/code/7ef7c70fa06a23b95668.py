"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: A refined Best-Fit that penalizes creating very small gaps (dead space).
    We want to fit items tightly, but avoid leaving a sliver that no future item can use.
    The L1 lower bound suggests we want to minimize total waste.
    
    Score = - (remaining_capacity_after_placement) + penalty_for_small_gap
    Actually, since higher score is better, we want to minimize the "badness" of the choice.
    Badness = remaining_capacity_after_placement.
    If the remaining capacity is very small (e.g., < 5), it's "dead" space.
    So we can penalize bins that result in a small remaining capacity by adding a large cost,
    or equivalently, reducing the score (since score is negative cost, we make it more negative? No, score is used for max).
    
    Let's define score such that max(score) is the best bin.
    Standard Best Fit: Minimize remaining space. So Score = -remaining_space.
    If remaining_space is small, -remaining_space is close to 0 (high).
    If remaining_space is large, -remaining_space is very negative (low).
    
    Problem: If we have a bin with 100 capacity, item is 96. Remaining = 4. Score = -4.
    If we have a bin with 100 capacity, item is 50. Remaining = 50. Score = -50.
    Best Fit picks the bin with remaining 4.
    
    But if the next item is 5, it fits in the 4? No.
    If the next item is 3, it fits.
    If the next item is 6, it doesn't fit in the 4, but might fit in the 50.
    
    The "dead space" penalty suggests: if remaining < threshold (e.g., 5), it's bad.
    We want to avoid bins that leave < 5 if possible, unless it's the only option or the best fit.
    
    Let's try:
    remaining = bins - item
    valid = remaining >= 0
    
    If remaining < 5, add a penalty to the cost (make score lower).
    Cost = remaining + penalty * max(0, threshold - remaining)
    Score = -Cost
    
    Let threshold = 5.
    If remaining = 4, Cost = 4 + 10*(5-4) = 14. Score = -14.
    If remaining = 6, Cost = 6 + 0 = 6. Score = -6.
    If remaining = 50, Cost = 50. Score = -50.
    
    So a bin leaving 6 is preferred over one leaving 4.
    A bin leaving 4 is preferred over one leaving 50.
    
    This effectively pushes the fit away from the "dead zone" if a slightly larger fit is available.
    
    However, we must ensure we only consider valid bins.
    Invalid bins should have a very low score (e.g., -inf).
    """
    remaining = bins - item
    valid = remaining >= 0
    
    # Initialize scores to a very low number for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    rem_valid = remaining[valid]
    
    # Threshold for "dead" space
    threshold = 5.0
    
    # Calculate cost
    # Base cost is the remaining space
    cost = rem_valid.copy()
    
    # Penalty for remaining space below threshold
    # We want to penalize small remainders.
    # Let's use a linear penalty: penalty_weight * (threshold - rem) for rem < threshold
    # If rem >= threshold, penalty is 0.
    small_gap = np.maximum(0, threshold - rem_valid)
    penalty_weight = 10.0  # Strong penalty to avoid small gaps
    cost += penalty_weight * small_gap
    
    # Score is negative cost
    scores[valid] = -cost
    
    return scores
# EVOLVE-BLOCK-END
