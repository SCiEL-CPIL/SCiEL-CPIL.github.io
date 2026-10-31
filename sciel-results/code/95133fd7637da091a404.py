"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We'll use a hybrid scoring strategy that adapts based on the current state.
# The idea:
# 1. For small items, we want to fill bins tightly (best-fit like).
# 2. For large items, we might want to save space for other large items (first-fit decreasing-like).
# 3. We also consider the "waste" or "fragmentation" of bins.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy:
    - Only consider bins with remaining capacity >= item.
    - Score based on a combination of:
      a) How well the item fits (minimize leftover space)
      b) But also consider the distribution of remaining spaces to avoid fragmentation
      c) For very large items, prefer bins that can accommodate them without wasting too much
    """
    # Start with a very low score for all bins
    scores = np.full(len(bins), -1e18)
    
    # Calculate remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # If item is very small relative to capacity, use best-fit (minimize leftover)
    # If item is large, we might want to be more selective
    
    # A good heuristic: score = -leftover space, but with a penalty for fragmentation
    # Actually, let's think about this differently.
    
    # The key insight: we want to minimize the number of bins used.
    # This is equivalent to maximizing the density of used bins.
    
    # Simple approach: prefer the bin where the item fits with the least waste
    # But this is essentially best-fit, which is what we have.
    
    # Let's try a different angle: consider the "opportunity cost" of using each bin.
    # If we use a bin that has a lot of remaining space, we might waste that space.
    # If we use a bin that has little remaining space, we fill it up more completely.
    
    # So the score should be: how full will the bin be after placing this item?
    # Score = (bins[i] - remaining[i]) / bins[i] = 1 - remaining[i]/bins[i]
    # But this requires knowing the original bin capacity, which we don't have directly.
    # Actually, bins[i] here is the REMAINING capacity, not the original.
    
    # Let me re-read the interface.
    # bins: np.ndarray of remaining capacity per bin.
    # So bins[i] is the remaining space in bin i.
    
    # So if we place item in bin i, the new remaining space is bins[i] - item.
    # The fraction of the bin that is now used = (original_capacity - (bins[i] - item)) / original_capacity
    # But we don't know original_capacity.
    
    # Alternative: just minimize the remaining space after placement.
    # That's best-fit: score = -(bins[i] - item)
    
    # Let's try to improve on best-fit by considering other factors.
    
    # Idea: if there are multiple bins with similar remaining space, prefer the one that
    # leads to a more balanced distribution of remaining spaces (to avoid fragmentation).
    
    # But this is complex to compute efficiently.
    
    # Let's try a simpler improvement: weight the score by the item size.
    # For small items, best-fit is good. For large items, we might want first-fit.
    
    # Actually, let's try: score = -(bins[i] - item) + epsilon * bins[i]
    # This slightly prefers bins with more remaining space (first-fit), which might be good
    # for large items to avoid creating many nearly-empty bins.
    
    # The epsilon should be small enough that best-fit still dominates for small differences.
    
    # Let's use a fixed epsilon. But what value?
    # Capacity is 100, so items are probably between 1 and 50 or so.
    # Let's try epsilon = 0.01
    
    scores[valid_indices] = -valid_remaining + 0.01 * bins[valid_indices]
    
    return scores
# EVOLVE-BLOCK-END
