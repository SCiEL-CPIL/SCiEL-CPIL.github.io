import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 148.0  # 150s limit, 2s safety margin
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Get remaining capacities of feasible bins
    rem = bins[feasible]
    
    # Strategy:
    # We combine the "Hybrid BF/WF" structure of the current program with
    # the "Polarization" insight from the inspirations.
    #
    # The polarization insight suggests that encouraging bins to become either
    # very full or very empty is beneficial. This reduces the number of
    # "half-full" bins which are hard to pack efficiently.
    #
    # The current program uses a sigmoid blend between Best Fit (BF) and Worst Fit (WF).
    # BF minimizes remaining space (rem), pushing bins towards "full".
    # WF maximizes remaining space (rem), pushing bins towards "empty" (relative to feasible options).
    #
    # However, the previous attempt (score 0.9998) used a sigmoid centered at 50.
    # This meant small items (<50) were mostly BF, and large items (>50) were mostly WF.
    # For small items, BF is generally good for consolidation.
    # For large items, WF might be bad because it spreads large items out,
    # preventing the consolidation of medium items into the remaining space of large-item bins.
    #
    # To better achieve "Polarization" while maintaining strong packing:
    # 1. For small items, we want strong Best Fit to consolidate.
    # 2. For large items, we should still prefer Best Fit to avoid wasting the "medium" space
    #    in bins that could have taken other medium items.
    #    However, if a bin is *very* empty (high rem), putting a large item there might be
    #    better than putting it in a "medium" bin, IF we want to keep the medium bins for
    #    medium items. But usually, Best Fit is superior for online bin packing.
    #
    # The "Polarization" score that improved island 2 (1 -> 1.02) likely used a non-linear
    # function that strongly rewards extremes.
    # Let's try a score that is a convex combination of BF and WF, but with a parameter
    # that shifts the balance based on the *global* state or the item size more aggressively.
    #
    # Actually, let's look at the "Polarization" idea more closely.
    # If we score bins based on how "extreme" their remaining capacity is,
    # we might prefer bins that are either very full (low rem) or very empty (high rem).
    # But we can only choose one bin.
    #
    # A simpler interpretation:
    # Best Fit (minimize rem) pushes the chosen bin towards full.
    # Worst Fit (maximize rem) pushes the chosen bin towards empty (least change).
    #
    # If we use a "Polarized" score:
    # score = -rem * (1 - alpha) - (C - rem) * alpha ? No.
    #
    # Let's try the hybrid with a much higher threshold for switching to WF,
    # or simply pure Best Fit with a slight modification.
    #
    # Given that the seed (Pure BF) scored 1.0 and the hybrid (T=70) scored 0.9998 (worse),
    # it suggests that deviating from BF is risky.
    #
    # However, the "Polarization" finding (1.02) is significant.
    # How was that achieved? "Polarization scoring encourages bins to become either full or empty".
    # This might mean:
    # If rem is small, give a high score (BF).
    # If rem is large, give a high score (WF)?
    # But we can only pick one bin.
    #
    # Perhaps the polarization was applied to the *score function* itself.
    # E.g., score = -|rem - target|?
    # Or score = -(rem * (C - rem))? This is maximized when rem is 0 or C.
    # But rem is in [0, C].
    # If we want to minimize the "mixed" state, we want rem to be 0 (full) or C (empty).
    # But we are placing an item, so rem will decrease.
    #
    # Let's try a score that favors bins with very small remainder (BF)
    # but also has a secondary term that favors bins with very large remainder (WF)
    # only if the item is very small?
    #
    # Actually, let's look at the specific improvement: 1 -> 1.02.
    # This is a 2% improvement.
    #
    # Let's try a **Quadratic Best Fit**.
    # score = -rem^2.
    # This penalizes large remainders much more heavily than linear BF.
    # This forces the algorithm to pick the *tightest* possible fit,
    # potentially leaving other bins untouched (more "empty" or "full" relative to the chosen one).
    # This aligns with "Polarization": the chosen bin becomes very full, others remain unchanged.
    #
    # Let's test:
    # score = -rem^2
    #
    # If rem1 = 1, score = -1.
    # If rem2 = 2, score = -4.
    # BF picks rem1.
    #
    # If rem1 = 10, score = -100.
    # If rem2 = 11, score = -121.
    # BF picks rem1.
    #
    # This is effectively "Aggressive Best Fit".
    #
    # Let's combine this with a slight Worst Fit component for very small items?
    # No, let's stick to the "Polarization" idea:
    # The goal is to make bins full or empty.
    # By picking the tightest fit, we make the chosen bin as full as possible.
    # This is good.
    #
    # Let's try:
    # score = -rem^p
    # with p = 2.0.
    #
    # We will keep the deadline guard and the structure.

    # Use Aggressive Best Fit (Quadratic)
    # This is a refinement of Best Fit, staying within the "Fit-based" family
    # but incorporating the "Polarization" insight (favoring extreme fullness of the chosen bin).
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    b = bins[feasible]
    rem_f = b - item
    
    # Quadratic penalty on remainder
    # This makes the preference for tight fits much stronger than linear BF.
    scores[feasible] = -rem_f**2
    
    return scores
