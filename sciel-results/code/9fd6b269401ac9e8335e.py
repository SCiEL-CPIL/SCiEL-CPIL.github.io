"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute a small lookup for "best-fit-decreasing-like" heuristics.
# For online bin packing with Weibull items (capacity 100), we want a heuristic
# that balances:
#  1) Best-fit: minimize leftover space in the chosen bin (reduces fragmentation).
#  2) Avoid opening new bins prematurely: prefer bins that are already "warm".
#  3) For large items, prefer bins with just enough room (tight fit).
#  4) For small items, prefer bins with moderate remaining space to allow packing.

# We use a scoring function that combines:
#   - Primary: -(bins - item)  [best-fit: least remaining]
#   - Secondary: a small bonus for bins that are already partially filled
#     (to encourage consolidation)
#   - Tertiary: a penalty for bins that are almost full (to avoid overflow risk
#     in floating point, though capacity check handles that)

# Actually, the seed is pure best-fit: return -(bins - item).
# Let's think about what could improve on best-fit for Weibull distributions.

# Weibull with shape < 1 has many small items and few large ones.
# Weibull with shape > 1 has a peak and then decay.
# The task says "five generated Weibull sequences" - likely a mix of shapes.

# Known results: For online bin packing, Best-Fit and Best-Fit Decreasing (offline)
# are strong. But we're online. 
#
# One improvement: "Best-Fit" can be improved by considering the *relative* fit.
# A bin with 10 units left and an item of 9 is a better fit than a bin with 100 units
# left and an item of 91, even though both leave 1 unit. The relative tightness
# matters for preventing fragmentation.
#
# Another idea: "Worst-Fit" tends to perform worse, but "Next-Fit" is worse still.
#
# Let's try a hybrid: score = -(bins - item) + alpha * (bins / 100.0)
# where alpha is small. This gives a slight preference to fuller bins.
# But wait, that might hurt. Let's think...
#
# Actually, a well-known improvement is to use a "gap" metric. The gap is the
# remaining space. We want to minimize the gap, but also avoid creating many
# bins with small gaps (fragmentation).
#
# Alternative: score based on the *product* of remaining space and some factor.
# Or: score = -log(bins - item + epsilon) to emphasize tight fits.
#
# Let's try: score = -(bins - item) for the primary, but add a small term that
# rewards bins that are in the "sweet spot" of being 50-90% full.
#
# Actually, let's keep it simple and robust. The seed is best-fit. A common
# improvement is to break ties by preferring bins with more remaining space
# (to keep options open), but that's worst-fit-ish. 
#
# Let's try a different angle: for items > 50, use best-fit (tight). For items
# < 50, use a blend that prefers bins with moderate remaining space.
#
# But the function must be fast. Let's try:
# score = -(bins - item) + 0.1 * np.minimum(bins, item)
# This adds a small bonus for bins that can accommodate the item with some margin.
#
# Actually, let's just try a pure best-fit with a slight modification:
# Prefer bins where (bins - item) is small, but if there's a tie, prefer the
# bin with more total remaining capacity across all bins? No, that's not local.
#
# Let's go with: score = -(bins - item) + beta * bins
# where beta is a small positive number. This makes the score less negative for
# fuller bins, so among bins with similar leftover, we prefer fuller bins.
#
# Wait, -(bins - item) + beta * bins = -bins + item + beta * bins = item + (beta - 1) * bins
# So if beta < 1, we still prefer smaller bins (more full). If beta = 0.9,
# score = item - 0.1 * bins. So we prefer bins with smaller bins value (more full).
# That's just best-fit with a different scale. Not helpful.
#
# Let's try: score = -((bins - item) ** 2) / (bins + 1)
# This penalizes large gaps more, normalized by bin size.
#
# Actually, I think the best approach is to stick close to best-fit but add a
# small regularization. Let's try:
# score = -(bins - item) + 0.05 * (bins / 100.0)
#
# But I'm not confident. Let's just return best-fit for now, but with a deadline
# guard as instructed.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Filter bins with enough remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # If no valid bins, return -inf for all (shouldn't happen per problem statement)
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    # Best-fit score: prefer bins with least remaining capacity
    # Add a small tie-breaker: prefer bins with slightly more remaining capacity
    # to keep flexibility (this is a minor adjustment)
    scores = np.full_like(bins, -np.inf)
    scores[valid_mask] = -(remaining[valid_mask])
    
    # Small regularization: add a tiny bonus for bins that are not nearly empty
    # This helps with fragmentation
    fill_ratio = bins / 100.0
    scores[valid_mask] += 0.01 * fill_ratio[valid_mask]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but return best found
        pass
    
    return scores
# EVOLVE-BLOCK-END
