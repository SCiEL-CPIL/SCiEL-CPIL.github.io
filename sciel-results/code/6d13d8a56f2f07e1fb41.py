"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We precompute a lookup table for a non-linear score function.
# The idea: we want to prefer bins that leave a very small residual (best-fit)
# or bins that are nearly full (to avoid wasting capacity on future items).
# However, pure best-fit (min residual) is the baseline.
# A known improvement in bin packing is to penalize "wasted" space more heavily
# when the residual is "medium", preserving medium-sized bins for medium items.
#
# Let's define score = -f(residual)
# We want high score for good bins.
# Good bins: small residual (best fit) or perhaps large residual? No, large residual is bad.
# But sometimes "Worst Fit" (largest residual) is better for certain distributions.
# For Weibull (which is often skewed), Best-Fit is generally strong.
#
# Let's try a scoring function that strongly prefers small residuals,
# but with a twist: if the residual is 0, it's perfect.
# If the residual is small, it's good.
# If the residual is large, it's bad.
#
# Let's experiment with a power function: score = -(residual ** alpha)
# If alpha > 1, it penalizes large residuals more.
# If alpha < 1, it penalizes small residuals more (relative to large ones).
#
# Let's look at the "bimodal" hint.
# Bimodal residual scoring: prefer residuals near 0 OR near capacity?
# "Preserving bins with moderate space" -> this suggests we should NOT use bins with moderate space if we can avoid it?
# Or does it mean we should use bins with moderate space?
# "outperform pure best-fit by preserving bins with moderate space for future items"
# This implies that pure best-fit uses up the "tight" fits, leaving behind bins with moderate space that might not fit future items well?
# Or does it mean that by not always picking the absolute best fit, we save the "perfect" fits for items that need them?
#
# Actually, if we always pick the best fit, we fill bins tightly.
# If we pick a slightly worse fit, we leave a larger gap.
#
# Let's try a score that is non-linear.
# We will compute residuals = bins - item.
# We only consider valid bins (residual >= 0).
#
# Strategy:
# 1. Compute residuals.
# 2. For valid bins, calculate a score.
# 3. Return scores.
#
# Let's try: score = -residual / (residual + epsilon) ? No, that's just monotonic.
#
# How about: score = - (residual ** 0.5) ?
# This makes small residuals "less bad" than linear.
#
# Let's try a different approach:
# The L1 lower bound is sum(items) / capacity.
# We want to minimize the number of bins used.
#
# Let's implement a "Best Fit Decreasing" variant? No, items arrive online.
#
# Let's stick to the bimodal hint.
# Maybe: score = - (residual ** 2) ?
# This heavily penalizes large residuals.
#
# Let's try:
# score = -residual
# is the baseline.
#
# Let's try:
# score = - (residual ** 1.5)
#
# Or perhaps a lookup table based on the expected item distribution?
# We don't know the distribution at runtime, but we know it's Weibull with capacity 100.
# Typical Weibull shapes might generate items in [0, 100].
#
# Let's try a simple heuristic:
# If the residual is very small (< 5%), it's a great fit.
# If the residual is moderate, it's okay.
# If the residual is large, it's bad.
#
# Let's use a piecewise linear or polynomial score.
#
# Let's try:
# r = bins - item
# valid = r >= 0
# scores = np.where(valid, -(r ** 0.8), -np.inf)
#
# Let's test this mentally.
# If r=0, score=0.
# If r=1, score=-1.
# If r=10, score=-6.3.
# If r=50, score=-28.
#
# Baseline (r=10) is -10.
# Here, r=10 is -6.3, which is HIGHER (better) than -10.
# So this function prefers larger residuals?
# Wait.
# Higher score = better.
# Baseline: r=10 -> -10. r=1 -> -1.
# My function: r=10 -> -6.3. r=1 -> -1.
# -1 > -6.3. So it still prefers r=1.
# But the penalty for r=10 is less severe than linear.
# This means it's "more forgiving" of larger residuals.
# This is closer to "Worst Fit" behavior than "Best Fit".
#
# The hint says "Bimodal residual scoring (preferring residuals near 0 or near capacity) ... reducing total bin count".
# "Preferring residuals near capacity" means preferring LARGEST residuals?
# That would be Worst Fit.
# But Worst Fit is usually bad for minimizing bin count in general, unless the distribution is specific.
#
# Let's re-read: "preferring residuals near 0 or near capacity".
# If we prefer near 0 (Best Fit) AND near capacity (Worst Fit), that's bimodal.
# How to combine?
# score = - ( (r - C/2) ** 2 ) ?
# This would prefer r=0 and r=C (if C is capacity).
# But r cannot be C (since item > 0, r < C).
# So it would prefer r=0 and large r.
#
# Let's try:
# r = bins - item
# C = 100 (capacity)
# score = - ( (r - 0) ** 2 ) - ( (r - C) ** 2 ) ? No, that's complex.
#
# Let's try a simple polynomial that is high at 0 and high at C.
# f(r) = r * (C - r) ?
# This is 0 at r=0 and r=C.
# We want HIGH score for good bins.
# So score = - f(r) = - r * (C - r).
# At r=0, score=0.
# At r=C/2, score = - C^2/4 (very low).
# At r=C, score=0.
# This prefers r=0 and r=C.
# Since r < C (item > 0), it prefers r=0 and large r.
# This matches "preferring residuals near 0 or near capacity".
#
# Let's implement this.
# C = 100.0
# r = bins - item
# valid = r >= 0
# scores = np.where(valid, -r * (C - r), -np.inf)
#
# Let's check the sign.
# We want the highest scoring bin.
# If r=0, score=0.
# If r=50, score=-2500.
# If r=99, score=-99.
# So r=0 is best, then r=99, then r=50 is worst.
# This seems to match the hint.
#
# We need to be careful with the deadline.
# This calculation is O(N) and very fast.
#
# Let's add a deadline guard just in case, although this is O(N).
# The task says "Keep an EXPLICIT deadline guard".
#
# We will implement the function with the bimodal score.

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Compute residuals
    residuals = bins - item
    
    # Valid bins are those with non-negative residuals
    valid = residuals >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Calculate scores for valid bins
    # Score = -r * (C - r)
    # This is 0 at r=0 and r=C, and negative in between.
    # We want to maximize this score.
    r = residuals[valid]
    scores[valid] = -r * (CAPACITY - r)
    
    # Deadline check (should be instant)
    if time.time() - start_time > 115:
        # Fallback to best fit if somehow slow (unlikely)
        scores = np.where(valid, -residuals, -np.inf)
        
    return scores
# EVOLVE-BLOCK-END
