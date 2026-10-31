"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We cache some global statistics to help with the heuristic.
# In an online setting, we don't know the future, but we can assume items are
# drawn from a distribution. The problem statement mentions Weibull sequences.
# However, we don't have access to the generator.
# We can estimate the "average" item size or similar on the fly, but
# the function is called per item.
# Let's stick to local heuristics that are robust.

# The seed is Best-Fit (BF): minimize remaining capacity (bins - item).
# Score = -(bins - item). Higher score -> smaller remaining capacity.
#
# Variants to try:
# 1. First-Fit Decreasing is not possible (online).
# 2. Worst-Fit (WF): maximize remaining capacity. Score = bins - item.
# 3. Best-Fit Decreasing (BFD) approximation?
# 4. Harmonic?
# 5. A combination: prefer bins that are "medium" full?
#    Or prefer bins that are close to full but not too close?
#
# Let's analyze the "excess over L1 lower bound".
# L1 lower bound for bin packing is usually sum(items) / capacity + max_item / capacity?
# Actually, the lower bound on the number of bins is max(ceil(sum(items)/C), ceil(max_item/C)).
# The "excess" is (num_bins_used - LB) / LB.
# To minimize this, we want to minimize num_bins_used.
#
# Best-Fit is generally good for reducing the number of bins.
# However, Best-Fit can create many small fragments.
# Worst-Fit tends to keep bins balanced, which might be better for large items?
#
# Let's try a hybrid or a slight modification of Best-Fit.
# One idea: Penalize bins that are too full? No, Best-Fit already prefers full bins.
# What if we prefer bins that have a remaining capacity that is a multiple of the item?
# Unlikely to help generally.
#
# Let's look at the "Weibull" distribution. Weibull with shape < 1 has a high probability of small items and a long tail of large items.
# If shape > 1, it's more uniform or concentrated.
# Without knowing the shape, robust heuristics are best.
#
# A known strong heuristic is "Best-Fit with Debris" or similar, but that's complex.
# Let's try "Best-Fit" but with a tie-breaker or a slight bias.
#
# Another idea: "Next-Fit" is bad. "First-Fit" is decent.
#
# Let's try a score that balances "fit" and "fragmentation".
# If we pack an item into a bin, the remaining space is R = bins[i] - item.
# If R is very small, it's "debris" and might be useless for future items (unless future items are very small).
# If R is large, we might be wasting space.
#
# Best-Fit minimizes R.
# Worst-Fit maximizes R.
#
# For Weibull distributions, especially with shape < 1 (many small items), Best-Fit is usually strong because it packs small items tightly into the gaps.
# If shape > 1 (more uniform), Worst-Fit might be competitive because it spreads the load and avoids creating small gaps that can't be filled.
#
# Let's try a weighted sum or a product.
# Or simply stick with Best-Fit as it is the standard baseline and hard to beat without distribution knowledge.
#
# However, the prompt asks for a "stronger algorithm".
# Let's try "Best-Fit" but with a penalty for opening new bins?
# The function only sees current bins. It doesn't know if a bin is "new" or "old".
# But `bins` array contains the remaining capacity.
# A bin with remaining capacity == C (full capacity) is a new bin.
# A bin with remaining capacity < C is an existing bin.
#
# Heuristic: Prefer existing bins over new bins?
# If we put an item in a new bin, we increase the bin count by 1.
# If we put it in an existing bin, we don't.
# So, generally, we should prefer existing bins if they fit.
# Best-Fit does this implicitly because an existing bin will have remaining capacity < C, and a new bin has C.
# Since we minimize remaining capacity, existing bins (with smaller R) are preferred over new bins (with R=C) UNLESS the existing bin has very little space left and the item is small?
# Wait. If item is small, and existing bin has R=1, new bin has R=C.
# Best-Fit picks R=1. This is good, it fills the gap.
# If item is large, and existing bin has R=1, it doesn't fit.
# So Best-Fit naturally prefers filling gaps in existing bins.
#
# Is there a case where Best-Fit is bad?
# Yes, if it creates many tiny fragments that can never be used.
# "Best-Fit Decreasing" (BFD) is offline and optimal for many cases.
# Online, "Best-Fit" is asymptotically optimal for uniform distributions?
#
# Let's try a slight modification: "Best-Fit" but if the remaining capacity is extremely small (e.g., < 5% of capacity), maybe it's better to open a new bin to keep the existing bin's space "clean"?
# No, that's "First-Fit" like behavior.
#
# Let's look at the score definition again: "mean percent excess of bins used over the L1 lower bound".
# This is a global metric.
#
# Let's try a randomized approach? No, must be deterministic for reproducibility in evaluation usually, or at least stable.
#
# Let's try "Worst-Fit" (WF) and see if it's better for Weibull?
# For Weibull with shape < 1, WF is often worse than BF.
# For Weibull with shape > 1, WF can be better.
#
# Since we don't know the shape, maybe we can detect it?
# We can keep a history of item sizes and bin states.
#
# Let's implement a heuristic that adapts.
# Track the average item size.
# If items are getting smaller on average, use Best-Fit.
# If items are uniform/large, use Worst-Fit?
#
# Actually, a very strong heuristic is "Best-Fit" combined with a check on the "waste".
#
# Let's try this:
# Score = -(bins - item) + lambda * (bins - item)**2 ?
# This penalizes large remaining capacity more? No, Best-Fit minimizes R.
#
# What if we use a score that prefers bins where the remaining capacity is close to the item size?
# That's just Best-Fit.
#
# Let's try a different angle: The L1 lower bound is roughly sum(items)/C.
# To minimize bins, we want to maximize the sum of items packed per bin.
#
# Let's try "Best-Fit" but with a tie-breaking rule that prefers bins with higher remaining capacity?
# No, Best-Fit picks the min R. If there are ties, it doesn't matter much.
#
# Let's try "First-Fit" (FF).
# FF is often competitive and faster.
# Score: -index? No, we don't have indices in the `bins` array directly, but we can assume the array order is stable.
# If we return a score that favors lower indices?
# `bins` is an array. If we do `score = -bins + item`, that's BF.
# If we do `score = -np.arange(len(bins))`, that's FF (picks first available).
#
# Let's compare BF and FF.
# BF is generally better for random inputs.
#
# Let's try a "Harmonic" strategy?
#
# Given the constraints and the "Weibull" mention, Best-Fit is the safest bet for improvement if the seed is just BF.
# Wait, the seed IS Best-Fit.
# "CURRENT PROGRAM (score 1.0000): ... return -(bins - item)"
# So we are starting from BF.
#
# How to beat BF?
# 1. Adaptive Heuristic: Switch between BF and WF based on observed item sizes.
# 2. "Best-Fit with Debris": Avoid creating debris < threshold.
# 3. "Next-Fit Decreasing" approximation?
#
# Let's implement an adaptive strategy.
# We maintain a global state:
# - `item_history`: list of recent item sizes.
# - `bin_states`: list of remaining capacities? We get `bins` every time.
#
# We can calculate the "variance" of recent items.
# If variance is low (uniform), WF might be better? Or BF?
# For uniform distributions, BF and WF are similar, but BF is often slightly better for minimizing bins.
# For skewed distributions (Weibull < 1), BF is much better.
#
# Maybe the key is in the "debris".
# If we have a bin with R=2 and an item of size 1, BF puts it there. R becomes 1.
# If we have a bin with R=50 and an item of size 1, BF puts it in R=2.
# This is good.
#
# What if we avoid putting small items into bins that are already very full, to save the "perfect fit" for a larger item?
# No, larger items won't fit in R=2 anyway.
#
# Let's try a "Soft" Best-Fit.
# Score = -1 / (bins - item + epsilon)
# This penalizes very small remaining capacity more aggressively?
# If R=1, score = -1.
# If R=2, score = -0.5.
# If R=10, score = -0.1.
# This prefers R=1 over R=2. Same as BF.
#
# What if we use Score = -(bins - item) ** 0.5 ?
#
# Let's look at the "L1 lower bound" excess.
# Excess = (N_bins - LB) / LB.
#
# Let's try a simple adaptation:
# If the item is small (< 10% of capacity), use Best-Fit.
# If the item is large (> 50% of capacity), use Worst-Fit?
# Why? Large items should go into bins with the most space to avoid blocking smaller items?
# Actually, large items should go into the "best" fit to minimize waste?
# If I have a large item, I want to put it in a bin where it fits best?
# Or in a bin where it leaves the most space for others?
# If I put a large item in a bin with R=100 (new bin), I have 100-L left.
# If I put it in a bin with R=101 (existing), I have 1 left.
# BF picks R=101. This creates a 1-unit gap.
# WF picks R=100 (if it's the largest remaining).
#
# For large items, creating small gaps is bad because few small items will fit?
# Actually, small items are likely to fit in 1-unit gaps if they are very small.
# But if the distribution has a minimum item size, then 1-unit gaps are wasted.
#
# Let's assume the Weibull distribution has items > 0.
#
# Let's try a heuristic that penalizes "wasted space" in existing bins.
#
# Alternative: Use the "Next-Fit" logic for new bins?
#
# Let's try this specific improvement:
# **Best-Fit with a penalty for very small remaining capacity.**
# If remaining capacity after placement is < threshold (e.g., 5), it's likely to be debris.
# So, we might prefer a bin with slightly larger remaining capacity if it avoids creating debris.
#
# Score = -(bins - item) + penalty
# If (bins - item) < threshold:
#     penalty = -large_value  # Make score lower (worse)
#
# Let's define:
# R = bins - item
# If R < 5: score = -1000 (very bad)
# Else: score = -R
#
# This would avoid creating small fragments.
# Let's test this logic.
# Item=1, Bin1 R=10 (after: 9), Bin2 R=2 (after: 1).
# BF picks Bin2 (R=2 -> 1).
# With penalty: Bin2 creates R=1 (<5), so score is -1000. Bin1 creates R=9 (>5), score is -9.
# We pick Bin1.
# This leaves Bin2 with R=2.
# Is this better?
# If future items are size 1, Bin2 (R=2) can take one more. Bin1 (R=9) can take 9 more.
# We used Bin1 for the item. Bin1 now has 9. Bin2 has 2.
# Total remaining: 11.
# If we used Bin2: Bin2 has 1. Bin1 has 10. Total remaining: 11.
# Same total. But Bin2's 1 is likely useless. Bin1's 10 is useful.
# So avoiding debris is good.
#
# Let's implement this "Debris-Avoiding Best-Fit".
#
# We need to tune the threshold.
# Capacity is 100.
# Threshold = 5? 10?
# Let's try 5.
#
# Also, we need to handle the case where NO bin has R >= threshold.
# Then we must pick the best available (minimum R).
#
# Implementation:
# 1. Calculate R = bins - item for all bins where bins >= item.
# 2. For these bins, if R < threshold, assign a very low score (e.g., -1e6).
# 3. Otherwise, assign score = -R.
# 4. For bins where bins < item, score = -1e9 (invalid).
# 5. Return scores. The highest score is picked.
#
# Wait, if all valid bins have R < threshold, they all get -1e6.
# Then the highest score is -1e6. It doesn't matter which one is picked?
# We should still prefer the one with the smallest R among the "bad" ones?
# Or the largest R?
# If all create debris, we might as well create the smallest debris? (BF)
# Or the largest debris to keep it useful?
# If R=1 and R=2, both are <5.
# If we pick R=1, we create 0 debris? No, R is remaining.
# If item fits, new_R = R.
# If R=1, new_R=1. Debris=1.
# If R=2, new_R=2. Debris=2.
# 1 is smaller debris. So BF is better among the "bad" ones.
# So for R < threshold, score = -R (normal BF).
# For R >= threshold, score = -R + bonus?
# Or just:
# If R < threshold: score = -R
# Else: score = -R + 1000 (bonus for not creating debris)
#
# Let's refine:
# We want to prefer bins that result in R >= threshold.
# Among those, prefer smallest R (BF).
# Among those that result in R < threshold, prefer smallest R (BF).
#
# So:
# scores = -R
# valid_mask = bins >= item
# R = bins[valid_mask] - item
# debris_mask = R < threshold
# scores[valid_mask] = -R
# scores[valid_mask & ~debris_mask] += 1000  # Boost non-debris bins
#
# This will always pick a non-debris bin if one exists.
# If multiple non-debris bins exist, it picks the one with smallest R (highest -R + 1000).
# If no non-debris bins exist, it picks the one with smallest R (highest -R).
#
# This seems like a solid improvement over plain BF.
#
# Let's add a global variable to track if this is helping? No, we can't.
# Let's just implement it.
#
# Threshold choice:
# If items are Weibull, the minimum item size might be small.
# If min item is 0.1, then R=1 is not debris, it can fit 10 items.
# If min item is 1.0, then R=1 is debris (can fit 0 or 1 items).
#
# Without knowing the min item size, 5% of capacity (5.0) is a reasonable guess for "debris".
#
# Let's also consider the "new bin" case.
# If we open a new bin, R = 100 - item.
# If item is small (e.g., 1), R=99. This is non-debris. Score = -99 + 1000 = 901.
# If we put it in an existing bin with R=10, new R=9. Non-debris. Score = -9 + 1000 = 991.
# We prefer the existing bin. Correct.
#
# If existing bin has R=2, new R=1. Debris. Score = -1.
# New bin R=99. Non-debris. Score = 901.
# We prefer the new bin.
# This avoids creating debris in the existing bin.
# Is this good?
# We open a new bin, increasing bin count by 1.
# But we preserve the existing bin's R=2 for future small items.
# If we had put it in the existing bin, we would have R=1 (worse).
# So we avoided making the gap smaller.
# But we paid the cost of a new bin.
# If the item was small, maybe we should have put it in the existing bin to save the new bin?
#
# Example:
# Bin1: R=2. Item: 1.
# Option A (New Bin): Bin1 R=2, New Bin R=99. Total R=101. Bins+1.
# Option B (Bin1): Bin1 R=1. Total R=1. Bins same.
#
# If future items are size 1:
# Option A: Bin1 takes 1 (R=1). New Bin takes 1 (R=98).
# Option B: Bin1 takes 0 (R=1 is too small? No, 1 fits 1). Bin1 R=0.
#
# It depends on the future.
# Generally, saving a bin is good. But creating useless debris is bad.
#
# Maybe the bonus should be smaller?
# Or the threshold should be dynamic?
#
# Let's try a simpler modification first: **Best-Fit with a preference for "medium" fullness?**
# No, BF is standard.
#
# Let's stick with the **Debris-Avoiding Best-Fit** with a small bonus.
# Bonus = 10.
# Threshold = 5.
#
# If R >= 5: score = -R + 10
# If R < 5: score = -R
#
# Case: Bin1 R=10 (new R=9, score -9+10=1), Bin2 R=2 (new R=1, score -1).
# Pick Bin1.
#
# Case: Bin1 R=4 (new R=3, score -3), Bin2 R=5 (new R=4, score -4+10=6).
# Pick Bin2.
#
# This seems reasonable.
#
# Let's implement this.

DEBRIS_THRESHOLD = 5.0
DEBRIS_BONUS = 10.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    scores = np.full(len(bins), -1e9)
    
    # Find bins that can fit the item
    valid = bins >= item
    
    if not np.any(valid):
        # No bin can fit the item.
        # This should not happen if the evaluator ensures feasibility,
        # but if it does, we return -1e9 for all, or maybe the max bin?
        # The contract says "enough remaining capacity".
        # If none, we might need to open a new bin?
        # The function returns scores for EXISTING bins.
        # If no existing bin fits, the caller likely opens a new one.
        # We just return the scores.
        return scores
    
    R = bins[valid] - item
    
    # Base score: Best-Fit (minimize R)
    base_scores = -R
    
    # Apply debris bonus
    non_debris = R >= DEBRIS_THRESHOLD
    base_scores[non_debris] += DEBRIS_BONUS
    
    scores[valid] = base_scores
    
    return scores
# EVOLVE-BLOCK-END
