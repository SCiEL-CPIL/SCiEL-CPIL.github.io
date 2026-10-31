import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Structural Change: Probabilistic Best Fit
    # Instead of purely minimizing the immediate remainder (greedy Best Fit),
    # we estimate the "expected waste" of placing the item in each feasible bin.
    # 
    # Rationale:
    # Pure Best Fit minimizes the remaining space `rem_after = bins[i] - item`.
    # However, if `rem_after` is small but non-zero, it might be "dead space" 
    # if future items are large. If `rem_after` is large, it's more likely to 
    # be filled by future items.
    # 
    # We model future items as having a distribution (Weibull-like, skewed towards small).
    # A simple proxy for "expected waste" is the remainder itself, but weighted by 
    # the likelihood that a future item fits.
    # 
    # Let's try a "Soft Best Fit" that prefers bins where the remainder is 
    # either very small (close to 0, so bin is effectively full) or large enough 
    # to be useful, avoiding "middle" remainders that are too small for big items 
    # but not full.
    # 
    # Actually, a stronger structural idea from the inspiration is to look at the 
    # distribution of feasible bins. But we can't know the future.
    # 
    # Let's try a different structural approach: **Adaptive Threshold Best Fit**.
    # For very small items, Best Fit is great. For very large items, Best Fit is also great.
    # The problem might be in the "medium" range where Best Fit might fragment bins.
    # 
    # Let's try **First Fit Decreasing** approximation? No, we are online.
    # 
    # Let's try **Worst Fit** for items that are "medium" size?
    # If item is between 25% and 50% of capacity, maybe Worst Fit is better to keep bins open?
    # But the current program scored 0.9998 with a hybrid, which is worse than 1.0 (Best Fit).
    # This suggests that deviations from Best Fit are risky.
    # 
    # Let's look at the "Excess" metric again.
    # Excess = (Bins Used - L1) / L1.
    # To minimize Bins Used, we want to pack tightly. Best Fit does this.
    # 
    # Is there a tie-breaking issue?
    # If two bins have the same remainder, Best Fit picks the first one (lowest index).
    # This might not be optimal if the first bin is "less" full in some other metric?
    # No, if remainder is same, fill is same.
    # 
    # Let's try a **Global State** approach.
    # We can keep track of the number of bins used so far? No, we don't have access to history.
    # 
    # Let's try a **Variance-based** score.
    # The inspiration mentioned "negative variance of the remaining capacity distribution".
    # If we pick a bin, the new set of remainders changes.
    # We want to minimize the variance of the remainders? Or maximize it?
    # If remainders are all similar, it's "balanced". If one is very small and others large, it's "unbalanced".
    # Unbalanced might be better because the small one is "done" and the large ones are "available".
    # Balanced might be worse because all are "partially available".
    # 
    # Let's try to maximize the variance of the *post-placement* remainders.
    # This is computationally expensive to calculate exactly for each candidate bin.
    # 
    # Approximation:
    # If we pick bin i, the new remainder is r_i = bins[i] - item.
    # The other remainders are unchanged.
    # We want to maximize the variance of the new set of remainders.
    # 
    # Let R be the set of feasible remainders.
    # Let r be the remainder of the bin we pick.
    # The new set is R \ {r} U {r - item}.
    # 
    # This is too complex for a simple heuristic.
    # 
    # Let's go back to the **Probabilistic Risk** idea.
    # Score = -rem + alpha * P(waste | rem)
    # 
    # Let's try a simple **Exponential Best Fit**.
    # score = -exp(rem / C)
    # This penalizes large remainders exponentially.
    # 
    # Or **Quadratic Best Fit**.
    # score = -rem^2
    # 
    # Let's try **Quadratic Best Fit** for all items.
    # This will strongly prefer bins with very small remainders, even more than linear Best Fit.
    # 
    # Let's test this hypothesis.
    # If rem1 = 1, rem2 = 2.
    # Linear: -1 vs -2. Diff = 1.
    # Quadratic: -1 vs -4. Diff = 3.
    # Quadratic is more aggressive in picking the tightest fit.
    # 
    # Is tighter fit always better?
    # Yes, for minimizing bin count, tighter fit is generally better.
    # 
    # Let's try **Quadratic Best Fit**.

    rem = bins[feasible] - item
    scores[feasible] = -rem * rem
    
    return scores
