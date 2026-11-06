"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an arriving item. Higher score is preferred.
    The evaluator places the item in the highest-scoring bin among those with enough capacity.
    
    Strategy: A refined Best-Fit with a small penalty for bins that are already very full
    (to avoid creating many small scraps) but primarily prefers bins where the item fits best.
    
    We use a smooth approximation of the "best fit" logic.
    Let R = bins - item (remaining capacity after placing item).
    We want to minimize R.
    However, if R is very small, it's good. If R is large, it's bad.
    
    A common improvement over simple Best-Fit is to consider the "waste" created.
    Waste = R.
    So we want to minimize R.
    Score = -R.
    
    Let's try a slightly more sophisticated heuristic:
    1. Filter bins with capacity >= item.
    2. Among those, prefer the one with the smallest remaining capacity (Best Fit).
    3. However, if multiple bins have similar remaining capacity, prefer the one with larger total capacity? No, that doesn't help much.
    4. Alternative: Next Fit Decreasing is offline. We are online.
    
    Let's look at the L1 lower bound context.
    L1 lower bound is roughly sum(items)/capacity + max_item/capacity.
    To minimize excess, we want to pack tightly.
    
    Best Fit (minimize remaining) is generally strong.
    Let's try to add a tiny tie-breaker or smoothing.
    
    Consider the "Worst Fit" (maximize remaining) which spreads load. Usually worse for bin packing.
    Consider "First Fit" which is order-dependent.
    
    Let's stick to Best Fit but handle the "enough remaining capacity" constraint explicitly by setting invalid bins to -inf.
    
    Current seed: `return -(bins - item)`
    This returns negative remaining capacity.
    If bins[i] < item, then bins[i] - item is negative, so -(bins[i] - item) is positive.
    Wait, the evaluator says: "scores each bin with enough remaining capacity".
    Does the evaluator filter first?
    "scores each bin with enough remaining capacity; each arriving item goes to the highest-scoring bin."
    This implies the evaluator might filter bins where bins >= item before calling priority, OR it expects priority to return -inf for invalid bins.
    The interface description says: "np.ndarray of one score per bin with enough remaining capacity".
    This phrasing is slightly ambiguous. Does it mean the input `bins` array only contains valid bins?
    "priority(item: float, bins: np.ndarray) -> np.ndarray of one score per bin with enough remaining capacity"
    Usually in these frameworks, `bins` is the array of current free capacities for ALL bins.
    If the evaluator filters, `bins` would only have valid ones.
    If the evaluator does NOT filter, we must return -inf for invalid ones.
    
    Let's assume the standard case: `bins` contains all bins. We must ensure invalid bins are not chosen.
    The seed `return -(bins - item)` gives:
    If bins[i] >= item, score is negative (or zero).
    If bins[i] < item, score is positive.
    Since we want the HIGHEST score, and positive > negative, the seed would pick an INVALID bin if one exists!
    Unless the evaluator filters.
    
    Let's re-read carefully: "scores each bin with enough remaining capacity".
    This suggests the function is only called for valid bins, OR the function must output scores such that invalid bins are not selected.
    Given "FunSearch/EoH contract", typically the function returns a score for every bin in the input array.
    If the seed is `-(bins - item)`, and it scores 1.0 (baseline), it implies the seed is valid and working.
    If the seed picks invalid bins, it would crash or produce garbage.
    Therefore, it is highly likely that the evaluator filters `bins` to only include those with `bins >= item` before calling `priority`, OR the evaluator handles the filtering and `bins` passed in are already valid.
    
    HOWEVER, looking at the code `return -(bins - item)`, if `bins` contains invalid bins, the score is positive. If valid bins have negative scores, the max would be an invalid bin.
    If the evaluator then tries to place an item in a bin that doesn't have enough capacity, it would fail.
    Since the seed scores 1.0 (valid), the evaluator MUST be filtering the bins passed to `priority`, or the evaluator ignores scores for invalid bins (which contradicts "goes to highest-scoring bin").
    
    Let's assume the evaluator passes ALL bins. To be safe, we should explicitly set scores to -inf for bins where `bins < item`.
    
    Revised Strategy:
    1. Calculate remaining = bins - item.
    2. Identify valid bins: remaining >= 0.
    3. For invalid bins, score = -inf.
    4. For valid bins, use a heuristic.
    
    Heuristic: Best Fit is `min(remaining)`. So score = `-remaining`.
    Is there a better heuristic?
    "Mean percent excess of bins used over the L1 lower bound".
    
    Let's try a "Best Fit with tie-breaking by larger bin size" or similar?
    Actually, a known improvement for online bin packing is to use a "smoothed" best fit or to consider the variance.
    
    Let's try: Score = -remaining - epsilon * bins.
    This prefers bins that are just big enough (small remaining) AND smaller total capacity?
    No, if we have two bins:
    Bin A: cap 100, item 10. Rem 90.
    Bin B: cap 110, item 10. Rem 100.
    Best Fit picks A (Rem 90).
    If we penalize by `bins`:
    Score A = -90 - 0.001*100 = -90.1
    Score B = -100 - 0.001*110 = -100.11
    A is still better.
    
    What if we prefer to fill larger bins?
    Score = -remaining + epsilon * bins.
    Score A = -90 + 0.001*100 = -89.9
    Score B = -100 + 0.001*110 = -99.89
    A is still better.
    
    Let's look at "First Fit Decreasing" insights. It's offline.
    
    Let's try a "Worst Fit" variant? No, usually worse.
    
    Let's try a "Next Fit" variant? No.
    
    Let's stick to Best Fit but add the safety check for invalid bins.
    
    Code:
    """
    remaining = bins - item
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    valid_mask = remaining >= 0
    # Best Fit: minimize remaining
    # Score = -remaining
    scores[valid_mask] = -remaining[valid_mask]
    
    # Tie-breaking?
    # If remaining is equal, which bin to pick?
    # If we have bins [10, 10] and item 5. Rem [5, 5].
    # It doesn't matter much.
    
    # Let's try a slight variation:
    # Sometimes, leaving a tiny bit of space is bad if it can't fit anything else.
    # But we don't know future items.
    
    # Let's try "Best Fit Decreasing" logic isn't applicable online.
    
    # What about a penalty for bins that are almost full?
    # If remaining is very small, it's good.
    # If remaining is zero, it's perfect.
    
    # Let's try to improve the tie-breaking or add a small regularization.
    # Score = -remaining - 1e-6 * bins
    # This prefers smaller bins if remaining is equal?
    # If rem is equal, smaller bin means smaller total capacity.
    # This might help save larger bins for larger future items?
    # Actually, if rem is equal, e.g., Bin A (10) and Bin B (20), item 5.
    # Rem A = 5, Rem B = 15. Not equal.
    # If Bin A (10) and Bin B (10), item 5. Rem 5, 5.
    # Doesn't matter.
    
    # Let's consider the distribution of Weibull items.
    # Weibull with shape k. If k < 1, heavy tail (many small, few large).
    # If k > 1, light tail.
    # Capacity 100.
    
    # A common strong heuristic is "Best Fit".
    # Let's ensure we handle the -inf correctly.
    
    # One potential issue: if all bins are invalid, we return -inf everywhere.
    # The evaluator might crash or handle it. We assume there's always a valid bin or the evaluator handles it.
    
    # Let's add a small epsilon to avoid exact ties? No, numpy argmax handles ties by picking the first one.
    
    # Let's try a different heuristic: "First Fit" is O(N) scan. "Best Fit" is O(N) scan.
    # Since we return an array, the evaluator does argmax.
    
    # Let's try to incorporate a "balance" factor.
    # Score = -remaining
    # This is the seed (effectively, assuming filtering).
    
    # To improve, let's try "Worst Fit" with a cap? No.
    
    # Let's try "Best Fit" but with a penalty for creating very small fragments?
    # Fragment size = remaining.
    # We want to minimize remaining.
    
    # Maybe the issue is that Best Fit creates many bins with small remaining space that can't be used.
    # "Next Fit" opens a new bin when the current one doesn't fit.
    # "First Fit" uses the first bin that fits.
    # "Best Fit" uses the bin that fits best (least waste).
    
    # Let's try a hybrid:
    # If the best fit remaining is very small (< threshold), use it.
    # Otherwise, maybe use a larger bin to keep options open?
    # This is risky.
    
    # Let's try a smooth version of Best Fit.
    # Score = -log(remaining + epsilon)
    # This emphasizes small remainings more strongly.
    # If rem = 1, log(1) = 0, score = 0.
    # If rem = 10, log(10) = 2.3, score = -2.3.
    # If rem = 100, log(100) = 4.6, score = -4.6.
    # This prefers rem=1 over rem=10 much more than linear -rem would.
    # Linear: -1 vs -10.
    # Log: 0 vs -2.3.
    # This might be better at packing tight.
    
    # Let's try Score = -log(remaining + 1e-6)
    
    # Or Score = -1 / (remaining + 1e-6)
    # If rem=1, score=-1.
    # If rem=10, score=-0.1.
    # If rem=100, score=-0.01.
    # This heavily favors small remainings.
    
    # Let's test this hypothesis.
    # The L1 lower bound is tight when items are packed efficiently.
    # Minimizing waste (remaining) is key.
    # The 1/rem or log(rem) functions penalize large waste more severely.
    
    # Let's go with Score = -1 / (remaining + epsilon) for valid bins.
    
    eps = 1e-9
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        # Use 1/remaining to emphasize tight fits
        scores[valid] = -1.0 / (remaining[valid] + eps)
        
    return scores
# EVOLVE-BLOCK-END
