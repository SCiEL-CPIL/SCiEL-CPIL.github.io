"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a fluid-flow-inspired heuristic.
# The idea is to score bins based on how "efficiently" they use the remaining capacity
# relative to the item size.
#
# Let R = bins[i] - item (remaining capacity after placing item).
# Let C = 100 (capacity, though we can estimate it from max bin if needed, but 100 is given).
#
# The directive suggests: "prioritizing bins where the item constitutes a large fraction of the remaining capacity".
# This is slightly ambiguous. Does it mean item / bins[i] is large? Or item / (bins[i] - item) is large?
# "Ratio of their current load to the item size" -> load / item?
# "Item constitutes a large fraction of the remaining capacity" -> item / bins[i]?
#
# If we prefer bins where item / bins[i] is large, we are preferring bins that are already quite full (small bins[i]).
# This is similar to Best-Fit.
#
# However, the "fluid flow" analogy often suggests minimizing the "waste" or "inefficiency".
# A common metric is the "fill ratio" or how well the item fits into the gap.
#
# Let's look at the observed change note:
# "Inverting the hybrid heuristic to use Worst Fit for large items ... and Best Fit for small items ... improved score."
# This suggests that for large items, we should use Worst-Fit (maximize remaining space), and for small items, Best-Fit (minimize remaining space).
#
# Why?
# - Large items: If we put a large item in a bin with just enough space (Best-Fit), we create a small gap. If that gap is too small for future items, it's wasted. Worst-Fit puts it in a bin with lots of space, leaving a large gap that can accommodate other items.
# - Small items: We want to pack them tightly into existing gaps to avoid opening new bins. Best-Fit does this.
#
# So, a hybrid approach:
# If item is "large" (e.g., > 50% of capacity), use Worst-Fit logic (score = bins[i] - item, maximize).
# If item is "small" (e.g., <= 50% of capacity), use Best-Fit logic (score = -(bins[i] - item), minimize).
#
# Let's refine the threshold.
# Capacity C = 100.
# Threshold T = 50?
#
# Let's try a smooth transition or a simple switch.
# A simple switch is robust.
#
# Score for Best-Fit: -R
# Score for Worst-Fit: R
#
# We need to ensure that the scores are comparable if we mix them, but since we only pick the max score, and we are effectively choosing the strategy based on item size, we can just switch the scoring function.
#
# However, we must be careful: if we switch to Worst-Fit for large items, we are maximizing R.
# If we switch to Best-Fit for small items, we are minimizing R (maximizing -R).
#
# Let's implement this hybrid.
#
# Additionally, the directive mentions "fluid flow" and "ratio".
# Maybe we can use a score like:
# score = (item / bins[i]) * alpha - (bins[i] - item) * beta ?
#
# Let's stick to the hybrid BF/WF first as it has a measured improvement in the lineage.
#
# Threshold: 0.5 * C = 50.
#
# Let's also consider a "middle" range.
# For items between 25 and 50, maybe a blend?
# For now, a hard switch at 50 is a clear structural change from pure BF.
#
# Let's try:
# if item > 50:
#     scores = bins - item  # Worst Fit
# else:
#     scores = -(bins - item)  # Best Fit
#
# Wait, the scores need to be comparable if we were to mix, but we aren't. We are choosing the strategy.
# The function returns a score array. The evaluator picks the max.
# So if we use WF for large items, the max score will be the bin with the largest remaining space.
# If we use BF for small items, the max score will be the bin with the smallest remaining space.
#
# This seems correct.
#
# Let's add a small refinement:
# For Best-Fit, we might still want to avoid "debris" if it's extremely small?
# The previous program tried that and scored 0.719 (worse than 1.0).
# So maybe pure BF is better for small items.
#
# For Worst-Fit, pure WF might be good.
#
# Let's try this hybrid.
#
# One more consideration: The "fluid flow" suggestion.
# "scoring bins based on the ratio of their current load to the item size"
# Load = bins[i] - (bins[i] - item)? No, load is 100 - bins[i].
# Ratio = (100 - bins[i]) / item?
# This is high for full bins and small items.
#
# "prioritizing bins where the item constitutes a large fraction of the remaining capacity"
# item / bins[i] is large -> bins[i] is small -> Best-Fit.
#
# So the directive's "fluid flow" description actually sounds like Best-Fit for all items?
# But the observed change says WF for large items improved things.
#
# Let's trust the observed change more than the ambiguous directive text if they conflict.
# The observed change is a "cell finding" from a tool, which is "measured".
#
# So: Hybrid BF (small) / WF (large).
#
# Let's set the threshold at 50.
#
# We can also try a softer transition using a sigmoid or linear blend, but a hard switch is simpler and less prone to tuning errors.
#
# Implementation:
# 1. Identify valid bins (bins >= item).
# 2. Calculate R = bins - item.
# 3. If item > 50:
#      scores[valid] = R
#    Else:
#      scores[valid] = -R
# 4. Invalid bins get -inf (or a very low number).
#
# Let's use -1e9 for invalid.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9)
    
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    R = bins[valid] - item
    
    # Threshold for switching between Best-Fit and Worst-Fit
    # Capacity is 100.
    threshold = 50.0
    
    if item > threshold:
        # Large item: Use Worst-Fit (maximize remaining space)
        scores[valid] = R
    else:
        # Small item: Use Best-Fit (minimize remaining space)
        scores[valid] = -R
        
    return scores
# EVOLVE-BLOCK-END
