"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an online bin-packing heuristic.
    
    The item goes to the bin with the highest score.
    We use a Best-Fit-Decreasing-like heuristic adapted for online setting:
    - Prefer bins that minimize wasted space (best-fit)
    - But also consider future items: prefer bins that leave more 'flexible' space
      (i.e., not too small, not too large)
    
    Strategy: We want to minimize the number of bins opened. The L1 lower bound
    is sum(items) / capacity. To get close to this, we should pack tightly.
    
    A good heuristic: Best-Fit (minimize remaining capacity after placing item)
    but with a penalty for leaving very small gaps that can't fit future items.
    
    Actually, let's think about what makes bins "good":
    - A bin that is almost full is good (tight packing)
    - A bin that is half-full might be better than a nearly-empty one for large items
    - But we don't know future items
    
    Simple effective heuristic: Best-Fit (minimize remaining space)
    But we can improve by considering the "waste" more carefully.
    
    Let's use a weighted combination:
    1. Primary: minimize remaining capacity (best-fit)
    2. Secondary: prefer bins that are not too empty (to avoid fragmenting)
    
    Actually, for Weibull-distributed items, the distribution is skewed.
    Let's try: score = -(bins - item) + small_bonus_for_moderate_fill
    
    But simplest robust approach: Best-Fit with a twist.
    Score = -(remaining_capacity) is best-fit.
    
    Let's try a different angle: Next-Fit Decreasing isn't online.
    For online, Best-Fit and First-Fit are standard.
    
    Best-Fit: minimize (bins[i] - item) for bins[i] >= item
    This is equivalent to maximizing (bins[i] - item) negatively, i.e., score = -(bins - item)
    
    But we can do better by considering that we want to keep some bins "clean" for
    large items. So maybe: score = -(bins - item) + alpha * bins
    
    Wait, that would prefer larger bins, which is First-Fit-like.
    
    Let's try: score = -(bins - item) ** 2  (minimize squared waste)
    Or: score = -1 / (bins - item + epsilon)  (strongly prefer tight fits)
    
    Actually, for bin packing, Best-Fit is known to be quite good.
    Let's try a variant: Best-Fit with a small preference for bins that are
    not completely empty (to avoid wasting a bin on a small item when a
    partially-filled bin could take it).
    
    score = -(bins - item) + beta * min(bins, item)
    This rewards bins that have some content (min(bins, item) is larger for
    partially-filled bins).
    
    Let's set beta = 0.1 as a small bonus.
    """
    # Filter bins that can fit the item
    valid_mask = bins >= item
    
    # For invalid bins, assign very low score
    remaining = bins - item
    # Only consider valid bins
    scores = np.full_like(bins, -1e10, dtype=np.float64)
    
    if np.any(valid_mask):
        valid_remaining = remaining[valid_mask]
        valid_bins = bins[valid_mask]
        
        # Best-fit: minimize remaining capacity
        # Add a small bonus for bins that already have some content
        # This helps avoid fragmenting empty bins
        content_bonus = 0.1 * valid_bins
        
        # Score: negative remaining (prefer tight) + small bonus for content
        scores[valid_mask] = -valid_remaining + content_bonus
    
    return scores
# EVOLVE-BLOCK-END
