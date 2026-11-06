"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We cache a small amount of state to implement a hybrid heuristic.
# The function signature must remain priority(item, bins) -> scores.
# We use a module-level variable to track the "current" bins state if possible,
# but since we only get bins as an array, we can't easily track history.
# However, we can implement a more sophisticated scoring based on the distribution
# of remaining capacities, which is a known improvement over simple Best-Fit.

# A common strong heuristic is "Best Fit Decreasing" but we are online.
# "Harmonic" or "Worst Fit" variants can sometimes outperform Best Fit on specific distributions.
# For Weibull, which is often skewed, Best Fit is usually strong.
# Let's try a "Best Fit with slack penalty" or a "Hybrid" approach.
# One known improvement is to consider the "gap" created.
# If we put item in bin with remaining r, new remaining is r - item.
# We want to minimize the sum of squares of remaining capacities?
# Or minimize the maximum remaining capacity?

# Let's try a scoring that balances fitting tightly and not fragmenting too much.
# Score = -remaining (Best Fit)
# Alternative: Score = -remaining^2 (Favors bins where the item fits very tightly, i.e., remaining is small)
# Alternative: Score = -(remaining - item/2) (Shifted Best Fit)

# Let's analyze the L1 lower bound.
# L1 lower bound for bin packing is max(n, sum(items)/capacity).
# Excess = (bins_used - L1) / L1.
# Minimizing bins_used is the primary goal.

# A sophisticated heuristic: "Next Fit" is bad. "First Fit" is okay. "Best Fit" is good.
# "Worst Fit" can be bad.
# "Harmonic" is complex.

# Let's try to approximate a "global" best fit by looking at the distribution.
# If we have many bins with similar remaining capacity, Best Fit is good.
# If we have a few bins with large remaining capacity, maybe we should use them to avoid opening new bins?
# Actually, opening new bins is the cost. We want to fill existing bins as much as possible.
# This suggests Best Fit is indeed the right direction.

# Let's try to add a small "tie-breaker" or "smoothing".
# If two bins have the same remaining capacity, does it matter?
# In online packing, the order of bins might matter if the heuristic is stateful, but here it's stateless.

# Let's try "Best Fit" but with a slight preference for bins that are "fuller" in a relative sense?
# No, Best Fit is absolute remaining.

# What about "Least Remaining" (Best Fit) vs "Most Filled"?
# Most Filled = -remaining. Same thing.

# Let's try a "Soft Best Fit" using a non-linear function.
# e.g., score = -1 / (remaining + epsilon)
# This heavily penalizes large remaining capacities, pushing us to fill bins almost completely.
# This might lead to more fragmentation if items are very different in size, but for Weibull (similar scale), it might be good.

# Let's try: score = -remaining**2
# This makes the preference for tight fits much stronger.

# Another idea: "FIFO" (First In First Out) or "LIFO"? No, we don't have history.

# Let's stick to a refined Best Fit.
# The current program is `return -(bins - item)`.
# This is equivalent to maximizing the filled amount, or minimizing remaining.

# Let's try a "Hybrid" that considers the "waste" created.
# Waste = remaining - item.
# We want to minimize waste.
# But also, we want to avoid creating bins with very small remaining capacity that can't fit anything else?
# This is the "fragmentation" problem.
# However, since we don't know future items, we can't optimize for this directly.

# A known result is that Best Fit is asymptotically optimal for some distributions.
# Let's try to improve the constant factor.

# Idea: "Best Fit" with a "capacity" normalization?
# No, capacity is fixed at 100.

# Let's try: `return -(bins - item) + 0.1 * (bins - item)**2`?
# No, that would prefer larger remaining capacities if the square term dominates.
# We want to prefer smaller remaining capacities.
# So `return -(bins - item)` is linear.
# `return -(bins - item)**2` is quadratic, preferring smaller remaining capacities more strongly.

# Let's test the quadratic hypothesis mentally.
# If item = 10.
# Bin A: remaining 20. New remaining 10.
# Bin B: remaining 11. New remaining 1.
# Best Fit picks B (remaining 1 < 10).
# Quadratic: Score A = -100, Score B = -1. B is better.
# What if:
# Bin A: remaining 15. New remaining 5.
# Bin B: remaining 12. New remaining 2.
# Best Fit picks B.
# Quadratic: Score A = -25, Score B = -4. B is better.

# It seems quadratic always prefers the tighter fit more strongly.
# Is there a case where a slightly looser fit is better?
# If we have:
# Bin A: remaining 10.
# Bin B: remaining 9.
# Item 9.
# Best Fit picks B (remaining 0).
# If next item is 1, it fits in A (remaining 9).
# If we picked A (remaining 1), next item 1 fits in B (remaining 9).
# Symmetric.

# What if:
# Bin A: remaining 10.
# Bin B: remaining 10.
# Item 6.
# Best Fit picks either. Remaining 4.
# Next item 4. Fits.
# If we had a "Worst Fit" approach, we might pick a bin with remaining 100? No, we only consider bins with enough capacity.

# Let's consider "Worst Fit" (Most Remaining).
# This tends to keep bins "open" for larger items.
# For Weibull, if the scale is small, items are small, so Best Fit is good.
# If the scale is large, items are large, Worst Fit might be better?
# But the task says "five generated Weibull sequences". We don't know the parameters.
# Usually, "Best Fit" is a safer bet for general online bin packing.

# Let's try a "Smart Best Fit" that accounts for the number of bins.
# If we have many bins, we want to consolidate.
# If we have few bins, we want to be careful.

# Actually, let's look at the "L1 lower bound" again.
# L1 = max(N, Sum/Cap).
# If Sum/Cap is the bottleneck, we want to minimize Sum/Cap usage, i.e., pack tightly.
# If N is the bottleneck, we want to minimize N, i.e., use as few bins as possible.
# Both point to Best Fit.

# Let's try a small perturbation to break ties or add robustness.
# `return -(bins - item) + np.random.normal(0, 0.01)`?
# No, randomization might hurt consistency.

# Let's try the quadratic penalty:
# `return -(bins - item)**2`

# Or a hybrid:
# `return -(bins - item) - 0.01 * (bins - item)**2`

# Let's go with the quadratic penalty as it emphasizes tight packing.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Only consider bins with enough capacity
    # The problem statement says "scores each bin with enough remaining capacity".
    # Bins with insufficient capacity should probably have a score of -infinity or very low.
    # The current program `return -(bins - item)` returns positive values for insufficient bins,
    # which would make them the "highest scoring" (since we pick max).
    # Wait, "highest-scoring bin".
    # If score = -remaining, then smaller remaining -> more negative score?
    # No, if remaining = 1, score = -1.
    # If remaining = 10, score = -10.
    # Max score is -1. So it picks the bin with smallest remaining. Correct.
    
    # BUT, if remaining is negative (insufficient), score is positive.
    # e.g., remaining = -5, score = 5.
    # This is higher than -1.
    # So the current program is BROKEN for bins that are too small?
    # The prompt says: "scores each bin with enough remaining capacity".
    # It implies we should only score those, or score others so they are not picked.
    # "the item goes to the highest-scoring bin".
    # If we return positive scores for invalid bins, we will pick them.
    # We must ensure invalid bins have the lowest possible score.
    
    # Let's fix the masking.
    scores = np.full_like(remaining, -np.inf, dtype=float)
    valid = remaining >= 0
    scores[valid] = -(remaining[valid])
    
    # Apply quadratic penalty for valid bins
    # scores[valid] = -(remaining[valid]**2)
    
    # Let's try the quadratic version
    scores[valid] = -(remaining[valid]**2)
    
    return scores
# EVOLVE-BLOCK-END
