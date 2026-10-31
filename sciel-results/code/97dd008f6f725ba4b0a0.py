import numpy as np
import time

# Global deadline guard
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
    # We want to minimize the number of bins.
    # Best Fit (BF) is strong: minimize remaining space.
    # Worst Fit (WF) is weak: maximize remaining space.
    # 
    # The "probabilistic risk" idea suggests looking at the distribution.
    # However, a more direct improvement over pure BF is to avoid creating
    # "unusable" gaps. A gap is unusable if it's too small to fit any future item.
    # Since we don't know future items, we assume a distribution.
    # Weibull with shape k and scale lambda.
    # For small items, BF is good.
    # For large items, BF might fill a bin too tightly, leaving a tiny gap.
    # But actually, filling tightly is good because it reduces the total "waste".
    #
    # Let's look at the "Excess" metric: (Bins Used - L1) / L1.
    # L1 = ceil(Sum(items) / C).
    # To minimize Bins Used, we need to maximize the fill of each bin.
    #
    # A known strong heuristic is "Best Fit Decreasing" (BFD), but we are online.
    # Online Best Fit (BF) is asymptotically optimal for some distributions?
    # Actually, BF has a competitive ratio of ~1.7.
    # First Fit (FF) has a competitive ratio of ~1.7.
    #
    # Let's try a "Smoothed Best Fit".
    # Instead of just minimizing `rem`, we penalize bins that are *very* close to full
    # if the remainder is likely to be wasted? No, if remainder is small, it's good.
    #
    # What if we prefer bins that are *not* the *tightest* fit, to keep some slack?
    # This is "Worst Fit" again.
    #
    # Let's try a hybrid based on the *variance* of the feasible bins' remaining capacities.
    # If the feasible bins have low variance (all similar), pick the one that minimizes remainder (BF).
    # If high variance, maybe pick the median?
    #
    # Actually, the inspiration says: "score bins based on the negative variance of the remaining capacity distribution... rather than the local fit".
    # This implies a global statistic.
    #
    # Let's compute the mean and std of the feasible bins' remaining capacities.
    # Score = -(rem) + alpha * (rem - mean_rem) / std_rem ?
    # This would penalize bins that are far from the mean?
    #
    # Let's try a simpler, robust improvement:
    # **Best Fit with a small epsilon penalty for very tight fits?**
    # No, tight fits are good.
    #
    # Let's try **First Fit Decreasing** logic? We can't sort items.
    #
    # Let's try **Harmonic** heuristic approximation.
    # Harmonic bins items into classes [1/2, 2/3], [1/3, 1/2], etc.
    # We can approximate this by checking if the item fits in a "slot".
    #
    # Let's try a **Weighted Best Fit**:
    # score = -rem * (1 + beta * (rem / C))
    # This penalizes larger remainders more?
    # If beta > 0, larger remainders get more negative score (preferred? No, we want high score).
    # We want to maximize score.
    # Seed: score = -(bins - item) = -rem.
    # Max score = min rem.
    #
    # If we use score = -rem^2, we prefer even smaller remainders.
    # If we use score = -sqrt(rem), we are less sensitive to very small remainders.
    #
    # Let's try **score = -rem** (Best Fit) but with a tie-breaker.
    # If two bins have the same remainder, pick the one with the larger *original* capacity?
    # All bins have same capacity C.
    #
    # Let's try **score = -rem + epsilon * bins**
    # This is just Best Fit.
    #
    # What if we use **score = -rem / (rem + item)**?
    # This normalizes the remainder by the total bin size after placement.
    # If rem is small, score is close to 0.
    # If rem is large, score is close to -1.
    # So it still prefers small remainders.
    #
    # Let's try the **probabilistic risk** idea more concretely.
    # "Negative variance of the remaining capacity distribution".
    # Var(rem_feasible).
    # If we pick a bin, the new remainder is rem - item.
    # We want to minimize the *expected* future waste.
    #
    # A simpler proxy:
    # If the item is large (> C/2), it can only go into a bin with rem > C/2.
    # In this case, Best Fit is critical to avoid wasting space.
    # If the item is small, many bins are feasible.
    # Using Best Fit consolidates items, which is good.
    #
    # The previous failure was using Worst Fit for small items.
    # So, **Best Fit for all items** is a strong baseline.
    #
    # Can we do better than pure Best Fit?
    # One issue with Best Fit: it might fill a bin to 99%, leaving 1% waste.
    # If we had put it in a 90% full bin, we would have 19% waste, but the 99% bin is "done".
    # Actually, 99% is better.
    #
    # What if we use **Best Fit** but prefer the bin with the *smallest* index among ties?
    # This is deterministic.
    #
    # Let's try a **Dynamic Best Fit**:
    # score = -rem
    # But if rem < item * 0.1, add a small bonus? No, that's already good.
    #
    # Let's try **score = -rem + gamma * (C - bins)**
    # (C - bins) is the current fill.
    # This adds a small preference for bins that are *already* filled (consolidation).
    # But Best Fit already prefers bins with small rem, which implies high fill.
    # rem = C - fill - item.
    # Minimizing rem is equivalent to maximizing fill.
    # So Best Fit already does this.
    #
    # Let's try **score = -rem** (Best Fit) and see if we can improve the tie-breaking.
    # In NumPy, `argmax` picks the first occurrence.
    #
    # Let's try a **soft Best Fit** using a temperature parameter.
    # score = -rem / T
    # This doesn't change the argmax.
    #
    # Let's try **score = -exp(rem / C)**?
    # Exponential decay.
    #
    # Let's stick to **Best Fit** but ensure it's implemented efficiently.
    # The current program has a threshold. Let's remove the threshold and use Best Fit always.
    # The seed is Best Fit. The current program is a hybrid that scored 0.9998 (very close to seed 1.0).
    # Wait, the seed score is 1.0?
    # "search score is seed/value".
    # If seed value is V_seed, and candidate value is V_cand.
    # Score = V_seed / V_cand.
    # If Score = 1.0, then V_cand = V_seed.
    # If Score > 1.0, then V_cand < V_seed (better).
    #
    # The current program has score 0.9998. This is *worse* than 1.0.
    # So the hybrid is slightly worse than the seed.
    # Therefore, we should revert to **Best Fit** or find a better heuristic.
    #
    # Let's try **Best Fit** with a slight modification:
    # **Best Fit Decreasing (BFD) is offline.**
    # Online, **Best Fit** is standard.
    #
    # Let's try **First Fit**.
    # First Fit is often comparable to Best Fit.
    #
    # Let's try **Worst Fit** for large items?
    # If item > C/2, only bins with rem > C/2 are feasible.
    # If we use Best Fit, we put it in the bin with the smallest rem > C/2.
    # This leaves the largest rem for other large items? No, it leaves the *smallest* rem.
    # This is good because it keeps the bins with larger rem available for other large items?
    # Actually, if we have two large items, A and B.
    # Bin 1: rem 60. Bin 2: rem 90.
    # Item A: 50.
    # Best Fit: A goes to Bin 1 (rem 10). Bin 2 (rem 90) is available for B (if B <= 90).
    # Worst Fit: A goes to Bin 2 (rem 40). Bin 1 (rem 60) is available for B (if B <= 60).
    # If B is 70, Best Fit works (B in Bin 2). Worst Fit fails (B doesn't fit in Bin 1).
    # So **Best Fit is better for large items**.
    #
    # For small items:
    # Bin 1: rem 10. Bin 2: rem 90.
    # Item C: 5.
    # Best Fit: C goes to Bin 1 (rem 5). Bin 2 (rem 90) remains.
    # Worst Fit: C goes to Bin 2 (rem 85). Bin 1 (rem 10) remains.
    # If we have many small items, Best Fit consolidates them into Bin 1, closing it.
    # Worst Fit spreads them out, keeping many bins open.
    # Closing bins is good. So **Best Fit is better for small items**.
    #
    # Conclusion: **Best Fit** is likely the best simple heuristic.
    #
    # The seed is Best Fit.
    # The current program is a hybrid that is slightly worse.
    #
    # Let's try to improve Best Fit with a **tie-breaker** or a **smoothing**.
    #
    # Idea: **Best Fit with a penalty for creating a remainder that is a "bad" size**.
    # What is a "bad" size? A size that is unlikely to fit any future item.
    # If the remainder is < 10, it's likely bad (unless we have very small items).
    # If the remainder is > 50, it's likely good (can fit large items).
    #
    # So, if Best Fit leaves a remainder < 10, maybe we should prefer the next best fit?
    # This is **Best Fit with a threshold**.
    #
    # Let's try:
    # 1. Find the bin with the minimum remainder (Best Fit).
    # 2. If min_remainder < 10, check the second best fit.
    # 3. If the second best fit has a remainder between 10 and 20, maybe it's better?
    #
    # This is complex to implement efficiently in vectorized NumPy.
    #
    # Let's try a simpler **score function**:
    # score = -rem
    # But if rem < 10, add a penalty of 5.
    # score = -rem - 5 * (rem < 10)
    #
    # Let's test this hypothesis.
    #
    # Implementation:
    # rem = bins[feasible] - item
    # scores_feasible = -rem
    # penalty = 5.0 * (rem < 10.0)
    # scores_feasible -= penalty
    #
    # Let's try this.

    rem = bins[feasible] - item
    scores_feasible = -rem
    
    # Penalty for very small remainders
    # If the remainder is too small, it's likely wasted space.
    # Threshold: 5.0 or 10.0?
    # Let's try 5.0.
    penalty = 5.0 * (rem < 5.0)
    scores_feasible -= penalty
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    scores[feasible] = scores_feasible
    
    return scores
