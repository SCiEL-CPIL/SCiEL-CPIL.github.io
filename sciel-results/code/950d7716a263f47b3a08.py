"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Fragmentation-aware bimodal scoring.
# The current best (1.0204) uses a bimodal score: -r * (C - r).
# This prefers residuals near 0 (best fit) or near C (worst fit/empty).
# However, "worst fit" (large residual) can be bad if it creates a bin with
# a large, unusable fragment that no future items can fill efficiently.
# 
# The directive suggests a "fragmentation-aware" model that penalizes bins
# where the item is a moderate fraction (20-50%) of the remaining space.
# This is slightly different from the bimodal view. Let's refine the bimodal
# idea by making the "preference for large residuals" conditional or softened.
#
# Actually, let's look at the "fragmentation" hint more closely.
# If I put an item of size `item` into a bin with residual `r`, the new residual is `r`.
# The "fragment" is `r`.
# If `r` is small, it's good (tight fit).
# If `r` is large, it might be bad (wasted space), UNLESS that space is likely to be filled.
# In online packing with Weibull (often skewed towards small items), large fragments
# might be filled by many small items, or might remain empty if the distribution
# has few items large enough to fill them efficiently.
#
# The bimodal score `-r * (C - r)` effectively says:
# - r=0 is best (score 0).
# - r=C is best (score 0).
# - r=C/2 is worst (score -C^2/4).
# This implies that using an empty bin (r close to C) is as good as a perfect fit.
# This is essentially "Worst Fit" for large residuals and "Best Fit" for small ones.
#
# Let's try to modify this to be more "Best Fit" dominant, but with a softening
# for very large residuals to avoid premature fragmentation, or to preserve
# empty bins for potentially large items?
# No, usually you want to fill bins.
#
# Let's try a different structural change:
# Instead of just scoring based on `r`, let's score based on the *quality* of the fit
# relative to the item size.
#
# Consider the ratio `item / bins`.
# If `item` is close to `bins`, it's a great fit.
# If `item` is small compared to `bins`, it's a poor fit (leaves large fragment).
#
# Let's try: score = - (residual ** alpha)
# We established alpha=1 is best fit.
# The bimodal one is effectively a quadratic.
#
# Let's look at the "moderate fraction" penalty.
# If `item` is 20-50% of the bin capacity, and we use a bin with residual `r`,
# maybe we should penalize it if `r` is also in a "bad" range?
#
# Let's try a hybrid:
# Base score: Best Fit (-r).
# Adjustment: If the residual is very large (e.g., > 50), add a bonus (less negative)
# to encourage using empty bins? No, that's Worst Fit.
#
# Let's re-evaluate the bimodal result. It improved 1.0 -> 1.02.
# This suggests that avoiding "medium" residuals is beneficial.
# Why? Because medium residuals are hard to fill.
# Small residuals are filled by small items (which are abundant in Weibull).
# Large residuals (empty bins) are kept for large items (which are rare but exist).
# If we use a bin with a medium residual, we create a fragment that is too big for
# small items to fill efficiently (maybe) and too small for large items to fit.
#
# So the bimodal logic is sound.
# How to improve it?
# Maybe the "peak" at r=C is too strong?
# If we have an item of size 10, and two bins:
# Bin A: residual 10 (perfect fit, r=10). Score: -10 * 90 = -900.
# Bin B: residual 90 (empty bin, r=90). Score: -90 * 10 = -900.
# They are tied.
# If we pick Bin B, we leave Bin A with r=10.
# Next item size 10. Bin A is perfect.
# If we pick Bin A, we leave Bin B with r=90.
# Next item size 10. Bin B is not perfect (r=80).
# So picking the "empty" bin for a small item might be bad because it wastes the
# "perfect fit" opportunity of the tight bin?
# Wait, if Bin A has r=10 and item is 10, Bin A becomes full (r=0).
# If we pick Bin B (r=90), Bin B becomes r=80. Bin A remains r=10.
# Next item 10. Bin A is perfect.
# So in this case, picking Bin B (empty) was fine, and we still had Bin A for the next one.
#
# But what if the next item is 5?
# Bin A (r=10) -> r=5.
# Bin B (r=80) -> r=75.
#
# The bimodal score treats r=10 and r=90 equally for an item of size 10.
# Is that optimal?
#
# Let's try to bias slightly towards Best Fit for small items, and allow Worst Fit
# only for larger items?
#
# Or, let's adjust the bimodal function to be asymmetric.
# f(r) = -r * (C - r) is symmetric around C/2.
# Let's try: f(r) = -r * (C - r) - beta * r
# This adds a linear penalty, making small r more attractive (less negative score? No, more negative).
# Wait, we want to MAXIMIZE score.
# Score = -r(C-r).
# If we subtract beta*r, the score becomes lower for larger r.
# So it penalizes large residuals more. This makes it more "Best Fit".
#
# Let's try: Score = -r * (C - r) - 0.1 * r
# At r=10: -900 - 1 = -901.
# At r=90: -900 - 9 = -909.
# Now r=10 is preferred over r=90.
# This seems more sensible. We should prefer tight fits over empty bins,
# unless the tight fit is *very* tight?
#
# Let's try a different structural change:
# Use the item size to modulate the score.
# If item is small, prefer tight fits (Best Fit).
# If item is large, maybe prefer empty bins (to avoid fragmentation)?
#
# Let's try:
# score = -r * (C - r) - k * item * r
# If item is large, the penalty for large r is larger?
# No, if item is large, r is likely smaller (since r = bin - item).
#
# Let's stick to the refinement of the bimodal score.
# The current score is 1.0204.
# Let's try adding a small linear term to break the symmetry and favor Best Fit slightly.
#
# Score = -r * (C - r) - 0.05 * r
#
# Let's also consider the "fragmentation" hint about moderate fractions.
# If `item / bins` is between 0.2 and 0.5, maybe we should penalize the bin?
# This is hard to implement efficiently in a vectorized way without branching.
#
# Let's try the asymmetric bimodal score first.
# It's a simple structural tweak to the winning family.

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Compute residuals
    residuals = bins - item
    
    # Valid bins are those with non-negative residuals
    valid = residuals >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores

    # Calculate scores for valid bins
    r = residuals[valid]
    
    # Bimodal base: -r * (C - r)
    # Asymmetric penalty: - 0.1 * r
    # This makes small residuals (Best Fit) more attractive than large residuals (Worst Fit)
    # by breaking the symmetry of the parabola.
    scores[valid] = -r * (CAPACITY - r) - 0.1 * r
    
    # Add a tiny tie-breaker for lower indices
    indices = np.arange(len(bins))
    scores[valid] -= indices[valid] * 1e-12
    
    # Deadline check
    if time.time() - start_time > 115:
        # Fallback to simple best fit
        scores = np.where(valid, -residuals, -np.inf)
        
    return scores

# EVOLVE-BLOCK-END
