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
    # The seed program uses Best Fit: score = -(bins - item) = -rem.
    # This minimizes the remaining space after placement.
    #
    # The current program (score 0.9998) uses a hybrid:
    # w = sigmoid((item - 50) / 10)
    # score = (1-w) * (C - bins) + w * (bins - item)
    # Note: (C - bins) is the fill level. (bins - item) is the remaining space.
    # Wait, the current program code:
    # score_bf = C - b  (Fill)
    # score_wf = b - item (Remaining)
    # scores[feasible] = (1.0 - w) * score_bf + w * score_wf
    #
    # If item is small (w ~ 0.1): score ~ 0.9 * Fill + 0.1 * Rem.
    # Since Fill + Rem = C - item (constant for all feasible bins? No, C is constant, item is constant.
    # So Fill = C - b, Rem = b - item.
    # Fill + Rem = C - item.
    # So score = (1-w)(C - b) + w(b - item)
    # = (1-w)C - (1-w)b + wb - w*item
    # = (1-w)C - w*item + b * (w - (1-w))
    # = const + b * (2w - 1)
    #
    # If w < 0.5 (item < 50): coefficient of b is negative. Minimize b. -> Best Fit.
    # If w > 0.5 (item > 50): coefficient of b is positive. Maximize b. -> Worst Fit.
    #
    # So the current program is:
    # item < 50: Best Fit (slightly softened)
    # item > 50: Worst Fit (slightly softened)
    #
    # The seed program is:
    # score = -rem = -(b - item) = -b + item.
    # Maximize score -> Minimize b. -> Always Best Fit.
    #
    # The current program scored 0.9998, which is slightly worse than the seed (1.0).
    # This implies that for this specific Weibull distribution, switching to Worst Fit for large items (item > 50)
    # was slightly detrimental, or the softening for small items was detrimental.
    #
    # Why might Worst Fit for large items be bad?
    # If item is large (e.g., 60), only bins with rem >= 60 are feasible.
    # Worst Fit puts it in the bin with the most space.
    # This leaves the "tighter" bins (rem 60-70) for future large items?
    # Actually, if you have a bin with rem 65 and a bin with rem 95.
    # Item 60.
    # BF: Put in 65 (rem 5). 95 remains for future.
    # WF: Put in 95 (rem 35). 65 remains for future.
    # If next item is 30.
    # BF state: Bin 1 rem 5, Bin 2 rem 95. Item 30 fits in Bin 2 (rem 65).
    # WF state: Bin 1 rem 65, Bin 2 rem 35. Item 30 fits in Bin 1 (rem 35) or Bin 2 (rem 5).
    # Both seem okay.
    #
    # However, Best Fit is generally considered stronger for minimizing bin count in online settings
    # because it consolidates items. Worst Fit tends to spread items out, leading to more open bins.
    #
    # The inspiration suggests: "Best Fit with a small epsilon penalty for very tight fits?" No, tight fits are good.
    # "Soft Best Fit".
    #
    # Let's look at the "Smoothed Best Fit" idea.
    # Instead of a hard switch or linear blend, use a non-linear function that stays close to Best Fit
    # but allows for slight deviations if they are beneficial.
    #
    # Hypothesis: The seed (pure Best Fit) is very strong.
    # The current program deviates from Best Fit for large items (Worst Fit) and softens it for small items.
    # Since it's worse, we should revert closer to Best Fit.
    #
    # Can we improve on pure Best Fit?
    # One known issue with Best Fit is that it can create "unusable" gaps if the remainder is very small
    # but not zero. However, since we minimize remainder, this is already handled.
    #
    # Another idea: **Tie-breaking**.
    # In pure Best Fit, if multiple bins have the same minimum remainder, NumPy's argmax picks the first one.
    # This is arbitrary.
    # Is there a better tie-breaker?
    # Maybe prefer the bin with the *smallest* index? (Already done).
    # Maybe prefer the bin with the *largest* index?
    # Maybe prefer the bin that is *least* full? (i.e., if rem is same, b is same, so fill is same).
    #
    # Let's try a **Non-linear Best Fit**.
    # score = -rem^p
    # If p > 1, we penalize larger remainders more heavily. This makes Best Fit "more aggressive".
    # If p < 1, we are less sensitive to remainders.
    #
    # Let's try p = 1.5 or 2.0.
    # This effectively makes the function steeper near rem=0.
    #
    # Alternatively, use a **Logarithmic** score?
    # score = -log(rem + epsilon)
    # This compresses the range.
    #
    # Let's look at the "Probabilistic Risk" idea from inspiration 1.
    # "Score bins based on the negative variance of the remaining capacity distribution".
    # This is a global statistic.
    # If the remaining capacities are very diverse (high variance), maybe we should be more careful?
    # Or maybe we should pick the bin that brings the system closer to a "balanced" state?
    #
    # However, the directive says: "KEEP this design and investigate improvements within its family."
    # The current design is a hybrid BF/WF. The seed is pure BF.
    # The current design is worse.
    # So the "family" is likely "Fit-based heuristics".
    #
    # Let's try **Pure Best Fit** but with a **Tie-Breaker** that prefers bins that are *less* full?
    # No, if rem is same, fill is same.
    #
    # What if we use **Best Fit** but with a **Penalty for Fragmentation**?
    # Fragmentation is high if many bins have small remainders.
    # If we pick a bin, we reduce its remainder.
    # If we pick a bin with a large remainder, we keep the small-remainder bins untouched?
    # No, Best Fit picks the smallest remainder.
    #
    # Let's try **Worst Fit** for *very* large items (e.g., > 80) and **Best Fit** for the rest?
    # The current program switched at 50. Maybe 50 is too low.
    # If item > 80, it's very hard to place.
    # If we use Best Fit, we might waste a "medium" bin that could have taken a medium item.
    # If we use Worst Fit, we use the "largest" bin, preserving the "medium" bins.
    #
    # Let's test this hypothesis:
    # Switch to Worst Fit only for items > 75 or 80.
    #
    # Or, let's try **Best Fit** for all items, but with a **Smooth Transition** to Worst Fit for very large items.
    #
    # Let's modify the sigmoid center and width.
    # Center = 75, Width = 10.
    #
    # Also, let's consider the "Soft Min/Max" idea.
    # Instead of a hard switch, we are already doing a soft blend.
    # The issue might be the *range* of the blend.
    #
    # Let's try:
    # 1. Pure Best Fit (Seed).
    # 2. Best Fit with a slight preference for *larger* bins in case of ties? (Not possible with simple score).
    # 3. Best Fit with a non-linear penalty: score = -rem^1.5.
    #
    # Let's try **score = -rem** (Best Fit) as the base, but add a small term that encourages
    # using bins that are *not* the absolute smallest if the difference is negligible?
    # This is "Soft Best Fit".
    #
    # Actually, let's look at the **First Fit** heuristic.
    # First Fit is O(N) but very fast.
    # It often performs comparably to Best Fit.
    # But we are asked to combine strategies.
    #
    # Let's stick to the Fit-based family.
    # The current program is a blend.
    # The seed is pure BF.
    #
    # Let's try **Pure Best Fit** first, to ensure we don't lose the seed's performance.
    # Then, if we want to improve, we can try a slight deviation.
    #
    # However, the task asks to "combine the strategy of the current program with the strongest ideas of the inspirations".
    # Current: Hybrid BF/WF.
    # Inspiration: Deadline guard, Soft/Non-linear scores, Global stats.
    #
    # Let's implement a **Refined Hybrid**:
    # 1. Keep the deadline guard.
    # 2. Use a sigmoid blend, but shift the threshold to the right (e.g., 70 instead of 50)
    #    so that Best Fit is used for more items.
    # 3. Use a non-linear score function for the blend?
    #    Or just adjust the sigmoid parameters.
    #
    # Let's try Center = 70, Width = 15.
    # This means:
    # Item < 60: Mostly Best Fit.
    # Item > 80: Mostly Worst Fit.
    #
    # Why might this be better?
    # Large items (>80) are rare in Weibull distributions with mean ~50-60?
    # If they are rare, the impact of the heuristic on them is small.
    # But if they exist, Worst Fit might preserve medium bins.
    #
    # Let's also consider **Best Fit** with a **Tie-Breaker** based on **Index**.
    # If two bins have the same score, pick the one with the lower index? (Standard).
    #
    # Let's try a different refinement: **Adaptive Threshold**.
    # If the average fill of all bins is high, use Best Fit.
    # If the average fill is low, use Worst Fit?
    # No, low fill means we have lots of space. Best Fit consolidates, which is good.
    #
    # Let's go with the **Shifted Sigmoid** approach.
    #
    # Parameters:
    # T = 70.0
    # W = 15.0
    #
    # And we'll use the standard linear blend.
    #
    # Additionally, let's add a **Global Deadline Guard** as suggested by inspiration 1.
    #
    # One more idea: **Normalization**.
    # The scores for BF and WF have different scales?
    # score_bf = C - b (Range: [C - (C-item), C - item] = [item, C-item])
    # score_wf = b - item (Range: [0, C - item])
    # They are comparable.
    #
    # Let's implement this.

    T = 70.0
    width = 15.0
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    b = bins[feasible]
    
    # Calculate weight for Worst Fit vs Best Fit
    # w close to 1 -> Worst Fit, w close to 0 -> Best Fit
    t = (item - T) / width
    # Clip t to avoid overflow in exp
    t = np.clip(t, -10, 10)
    w = 1.0 / (1.0 + np.exp(-t))
    
    # Best Fit score: maximize (C - bins) -> minimize bins
    # Worst Fit score: maximize (bins - item) -> maximize bins
    score_bf = C - b
    score_wf = b - item
    
    # Blend
    scores[feasible] = (1.0 - w) * score_bf + w * score_wf
    
    return scores
