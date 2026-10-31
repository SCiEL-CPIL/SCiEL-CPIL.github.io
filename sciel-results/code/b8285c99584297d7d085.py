import numpy as np

# Cache for fast lookup
# Since we don't know the number of bins in advance, we use a fixed-size array
# and track the number of active bins. However, the input `bins` is an array of
# remaining capacities for *all* bins? Or just active ones?
# The seed `return -(bins - item)` implies `bins` is the array of remaining capacities.
# The length of `bins` is likely the total number of bins.
# We need to know the capacity C to calculate fill ratios.
# The task says "capacity 100". We assume C=100.

C = 100.0
MAX_BINS = 10000  # Safe upper bound for caching

# State:
# We need to know how many items of each size we've seen? No, that's not helpful for bin packing state.
# We need to know the distribution of *current* bin fills?
# The input `bins` gives us the current remaining capacities.
# We can compute the fill of each bin as C - bin_remaining.

# Strategy:
# 1. Calculate the "ideal" placement.
# 2. Use a heuristic that balances "Best Fit Decreasing" (BFD) and "Worst Fit Decreasing" (WFD)
#    but dynamically based on the current state of the bins.
#
# A known strong heuristic is "Next Fit Decreasing" (NFD) is bad.
# "Best Fit" (BF) is generally good for small items.
# "Worst Fit" (WF) is good for large items to keep bins open for large items.
#
# A more sophisticated approach:
# Score = -alpha * (remaining_after) + beta * (some other factor)
#
# Let's try a hybrid:
# If the item is large (> C/2), use Best Fit (minimize remaining).
# If the item is small (< C/2), use a "Balance" strategy.
#
# Actually, let's look at the "Excess" metric.
# Excess = (Bins Used - L1_Lower_Bound) / L1_Lower_Bound
# To minimize this, we want to minimize Bins Used.
#
# "Harmonic" or "FIFO" are simple.
# "Best Fit" is often competitive.
#
# Let's try a dynamic threshold.
# If the average fill of active bins is high, we are in a "tight" phase -> Best Fit.
# If the average fill is low, we are in a "loose" phase -> Worst Fit (to consolidate small items? No, Worst Fit spreads them out, which is bad for consolidation. Best Fit consolidates).
#
# Wait, Worst Fit puts items in the emptiest bin. This keeps many bins partially filled.
# Best Fit puts items in the fullest bin (that fits). This consolidates items.
# Consolidation is generally better for minimizing bin count.
#
# So, Best Fit is usually better than Worst Fit for minimizing bin count.
#
# However, if we use Best Fit for everything, we might fill a bin too much and leave a gap that no future item can fit, whereas a slightly less full bin might have been better.
#
# Let's try "Best Fit with a penalty for very small remainders".
# If the remaining capacity after placement is very small (< item_size * 0.1), maybe it's a "waste".
#
# Alternative: "First Fit" is often surprisingly good and fast.
#
# Let's try a scoring function that prefers bins where the remaining capacity is "compatible" with the item.
#
# Let's try:
# Score = - (bins - item)  # Best Fit
#
# But the current program uses a threshold.
#
# Let's try a different structural change: **Lookahead-free but state-aware**.
# We can't do lookahead easily in a stateless function unless we cache.
#
# Can we cache?
# The function is called sequentially.
# We can maintain a global state of the bins?
# No, the `bins` array is passed in. It IS the state.
#
# So we must derive the heuristic from `bins` and `item`.
#
# Idea: **Relative Best Fit**.
# Normalize the remaining capacity by the item size?
#
# Let's try a simple improvement over the current program:
# The current program switches between Best Fit and Worst Fit at 50%.
# Maybe the threshold is wrong.
#
# Let's try **Best Fit Always**.
# It's simple and often strong.
#
# Or **Next Fit**? No.
#
# Let's try **Harmonic** heuristic approximation?
#
# Let's try a **Penalized Best Fit**:
# score = -(bins - item)
# But if (bins - item) is very small, add a penalty? No, small remainder is good (tight fit).
#
# What if we prefer bins that are *already* partially filled, to encourage consolidation?
# That is exactly Best Fit.
#
# What if we prefer bins that are *empty* to avoid fragmentation?
# That is Worst Fit.
#
# There is a known result that **Best Fit** is very close to optimal for many distributions.
#
# Let's try a **Hybrid** based on the item size relative to the *median* remaining capacity of feasible bins.
#
# 1. Filter feasible bins.
# 2. If no feasible bins, return -inf (or let evaluator handle? No, must return array).
# 3. Calculate median remaining capacity of feasible bins.
# 4. If item > median_remaining / 2, use Best Fit.
# 5. Else, use Worst Fit?
#
# Let's try **Best Fit** with a twist:
# If the best fit leaves a remainder that is "too small" to be useful (e.g., < 10% of C), maybe the next best fit is better?
# This is "Best Fit Decreasing" logic but online.
#
# Let's try **First Fit** (lowest index feasible bin).
# It's O(1) if we can find it fast.
#
# Actually, let's look at the score of the current program: 0.0590.
# The seed is 1.0.
# The current program is *worse* than the seed (lower score is better, so 0.059 < 1.0 is better? No, "lower is better" means 0.059 is a *better* score than 1.0?
# Wait. "Score: mean excess over the L1 lower bound in percent".
# "The evaluator measures the raw value (lower is better)."
# "search score is seed/value".
# If seed value is V_seed and candidate value is V_cand.
# Score = V_seed / V_cand.
# If V_cand < V_seed, Score > 1.0.
# If V_cand > V_seed, Score < 1.0.
#
# The current program has score 0.0590.
# This implies V_cand is much LARGER than V_seed.
# V_seed / V_cand = 0.059 => V_cand = V_seed / 0.059 ≈ 16.9 * V_seed.
# So the current program is **16x worse** than the seed.
#
# The seed is `return -(bins - item)`, which is Best Fit.
# The current program uses a hybrid that apparently performs poorly.
#
# Therefore, **Best Fit** (the seed) is likely a strong baseline.
# We should probably stick close to Best Fit or find why the hybrid failed.
#
# The hybrid used Worst Fit for small items. Worst Fit is generally bad for minimizing bin count.
# So, let's try **Best Fit** but with a slight modification.
#
# What if we use **Best Fit** but break ties by preferring the bin with the *largest* remaining capacity?
# (i.e., if two bins have the same remaining capacity after placement, pick the one that was emptier before? No, if remaining is same, they were same before).
#
# What if we use **Best Fit** but with a penalty for bins that are *almost* full?
#
# Let's try **First Fit**.
#
# Or **Next Fit**?
#
# Let's try a **Dynamic Threshold** Best Fit.
#
# Actually, let's try **Worst Fit** for large items and **Best Fit** for small items?
# The current program did the opposite: Best Fit for large, Worst Fit for small.
#
# Logic:
# Large items: Hard to place. Use Best Fit to find a spot that fits tightly?
# Or use Worst Fit to keep bins open?
# If a large item goes into a half-full bin, it might block other large items.
# If it goes into an empty bin, it's safe.
# So **Worst Fit** might be better for large items?
#
# Small items: Easy to place. Use Best Fit to fill gaps.
# So **Best Fit** might be better for small items?
#
# The current program did:
# Item > 50: Best Fit
# Item <= 50: Worst Fit
#
# This is the opposite of the logic above.
# So let's try the **inverse**:
# Item > 50: Worst Fit
# Item <= 50: Best Fit
#
# Let's try this.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    C = 100.0
    T = 50.0
    
    feasible = bins >= item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(feasible):
        return scores
    
    if item > T:
        # Worst Fit: Maximize remaining capacity
        scores[feasible] = bins[feasible]
    else:
        # Best Fit: Minimize remaining capacity
        scores[feasible] = -(bins[feasible] - item)
        
    return scores
