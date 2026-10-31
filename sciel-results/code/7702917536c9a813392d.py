"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters inferred from the task description (5 sequences, 1000 items, cap 100)
# We assume a Weibull distribution with shape k and scale lambda such that items are in (0, 100]
# A common choice for bin packing benchmarks is k=1.5, lambda=20 (mean ~18) or similar.
# However, we don't know the exact distribution. The "L1 lower bound" is sum(items).
# The score is "mean percent excess of bins used over the L1 lower bound".
# Excess = (bins_used - L1_bound) / L1_bound * 100? Or (bins_used * capacity - total_size) / total_size?
# Usually, efficiency is Total_Size / (Bins_Used * Capacity).
# L1 lower bound on bins is ceil(Total_Size / Capacity).
# Let's assume the standard Weibull(2.0, 20.0) or similar often used in these tasks.
# If we don't know the distribution, we can use a heuristic that approximates best-fit decreasing
# or uses a surrogate for "future waste".

# A robust structural change: use a "waste-aware" scoring function.
# Instead of just minimizing remaining space (Best Fit), we penalize creating "dead space"
# that is too small for typical future items.
# If typical items are around mean μ, a remainder r < μ is likely to be wasted if no smaller item comes.
# But since items are random, a small remainder is actually good if it forces the next item to go elsewhere,
# preserving larger bins? No, Best Fit is generally strong.

# Let's look at the "opportunity cost" idea from the prompt's knowledge:
# "penalizes remainders that are too small to be useful for most future items"
# This suggests a score that favors remainders that are either very large (flexible) or very small (tight fit, but maybe not if it's dead space).
# Actually, Best Fit (minimizing remainder) is known to be near-optimal for many distributions.
# First Fit Decreasing is better for offline, but we are online.
# Best Fit is a strong baseline.

# Let's try a "Fit Decreasing" surrogate by estimating the next item's size? No, we only see current item.
# We can estimate the expected waste.
# Let's use a non-linear penalty on the remainder.
# If remainder is very small, it's a good fit.
# If remainder is medium, it might be wasted.
# If remainder is large, it's flexible.

# A common improvement over Best Fit is "Best Fit Decreasing" logic applied online by
# preferring bins that, after placement, leave a remainder that is "useful".
# Usefulness of remainder r: P(next item fits in r).
# If items are Weibull(k, λ), P(X <= r) = 1 - exp(-(r/λ)^k).
# We want to maximize the chance that the next item fits? No, we want to minimize total bins.
# Placing an item in a bin with remainder r leaves r.
# If r is small, the bin is "closed" effectively.
# If r is large, the bin stays "open".
# Keeping bins open is good if we can fill them.
# However, if we have many open bins, we might not close them.

# Let's try a scoring function that combines:
# 1. Tightness (smaller remainder is better, like Best Fit)
# 2. But penalize remainders that are "awkwardly" sized (e.g., too small to fit the median item, but too large to be ignored).

# Hypothesis: Best Fit is 1.0.
# Maybe "Worst Fit" (maximizing remainder) is worse?
# Maybe "First Fit" is worse?

# Let's try a "Smart Best Fit" that considers the distribution.
# We assume Weibull(2.0, 20.0) as a guess, but we can make it adaptive or use a generic shape.
# Since we can't tune constants easily, let's use a fixed shape parameter k=2.0 and scale λ=20.0 (mean ~18).
# Actually, if the capacity is 100, and items are Weibull, the scale might be different.
# Let's assume the mean is around 10-20.
# Let's use a score that is the negative of the "expected future waste".
# Expected waste of a bin with remainder r is ∫_0^r P(X > x) dx? No.
# The waste is r if no future item fits.
# Probability no future item fits in r is (1 - CDF(r))^N_remaining?
# This is complex.

# Simpler structural change: **Fit Gap**.
# Score = - (remainder - threshold) if remainder > threshold else -remainder?
# No.

# Let's try **Best Fit with a twist**:
# Prefer bins where the remainder is closest to the *expected next item size*?
# No, that's not right.

# Let's look at the "opportunity cost" finding:
# "Adding opportunity_cost term (λ*(1-S(r))) ... improved island 2"
# S(r) is the survival function P(X > r).
# 1-S(r) is CDF(r).
# So the term is proportional to CDF(r).
# This term *increases* the score (makes it less negative? or more positive?)
# The finding says "penalizes remainders that are too small".
# If r is small, CDF(r) is small. So 1-CDF(r) is large?
# Wait, the text says: "penalizes remainders that are too small to be useful... complementing dead_space term which penalizes medium remainders."
# And "opportunity_cost term (λ*(1-S(r)))".
# If S(r) = P(X>r), then 1-S(r) = P(X<=r).
# If r is small, P(X<=r) is small.
# If the term is added to the score, and we want to *penalize* small remainders, the score should decrease (become less attractive) for small r.
# So if score = base_score + opportunity_cost, and we want lower score for small r, then opportunity_cost should be negative for small r?
# Or is the score *maximized*? Yes, "item goes to highest-scoring bin".
# So we want high score for good bins.
# If small r is bad, score should be low.
# If CDF(r) is small for small r, and we add λ*CDF(r), the score is low.
# This matches "penalizes small remainders" (by giving them low score).
# So the term is + λ * CDF(r).
# But wait, Best Fit (minimize r) gives high score to small r (score = -r).
# So Best Fit *likes* small r.
# The new term *dislikes* small r.
# This creates a trade-off: very small r is good (tight fit) but bad (dead space).
# This suggests an optimal remainder size around the mode or median of the distribution.

# Let's implement:
# Score = -w1 * r + w2 * CDF(r)
# Where r is the remainder.
# We need to estimate CDF(r).
# Assume Weibull(k, λ).
# CDF(r) = 1 - exp(-(r/λ)^k).
# We need k and λ.
# If we don't know them, we can use a robust approximation.
# Or we can use a "soft" version.
# Let's assume k=2, λ=20 (guess).
# If the actual distribution is different, this might hurt.
# However, the task says "Weibull sequences".
# Common Weibull for bin packing: k=1.5, λ=20?
# Let's try to make it less sensitive to exact parameters by using a simple threshold.
# Or, use the "Best Fit" score but subtract a penalty for remainders < median.
# Median of Weibull(k, λ) = λ * (ln(2))^(1/k).
# For k=2, λ=20: median ≈ 20 * 0.83 = 16.6.
# Penalty if r < 16.6?

# Let's try a different structural approach: **Fit Decreasing Surrogate**.
# In offline FFD, we sort items. Online, we can't.
# But we can prioritize bins that are "closer to full" in a way that mimics FFD.
# Actually, Best Fit is already quite good.

# Let's try **Worst Fit** (maximize remainder).
# Score = bins - item (positive remainder).
# This keeps bins open. Often worse than Best Fit for packing density.

# Let's try **First Fit** logic?
# First Fit picks the first bin that fits.
# In our vectorized interface, we can't easily simulate "first" without knowing the bin order.
# But we can assume bins are indexed 0..N-1.
# We can give a small bonus to lower indices?
# Score = -remainder + small_bonus * (1 / (index + 1))?
# This might simulate First Fit.
# First Fit is often worse than Best Fit.

# Let's go with the **CDF-adjusted Best Fit**.
# Score = -r + α * CDF(r)
# We need to choose α.
# If α is large, we prefer remainders where CDF is high (large r).
# If α is small, we prefer small r.
# Best Fit is α=0.
# Let's try α=1.0 with normalized CDF?
# CDF is in [0,1]. r is in [0, 100].
# So α must be large to have an effect.
# Let's scale r.
# Score = -(r / 100) + α * CDF(r)
# Let's try α = 2.0.
# This means we care about the "usefulness" of the remainder as much as its size.

# We need to define CDF.
# Let's use a Weibull with k=2.0, λ=20.0.
# If the distribution is different, this is a risk.
# But it's a structural change from pure Best Fit.

# Alternative: **Hybrid**.
# Use Best Fit, but if the remainder is very small (< 5), penalize it?
# No, small remainder is usually good.

# Let's stick to the CDF idea.
# We'll use a fixed Weibull approximation.
# To be safe, we can make the CDF term weaker.

import math

def _weibull_cdf(x, k=2.0, lam=20.0):
    # x is remainder
    # CDF(x) = 1 - exp(-(x/lam)^k)
    # Clip x to [0, 100]
    x = np.clip(x, 0, 100)
    return 1.0 - np.exp(-((x / lam) ** k))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Filter out bins that don't fit
    mask = remaining >= 0
    scores = np.full_like(remaining, -1e9, dtype=float)
    
    if not np.any(mask):
        return scores
    
    r = remaining[mask]
    
    # Base score: Best Fit (minimize remainder)
    # Scale remainder to [0,1] for balance with CDF term
    base = -r / 100.0
    
    # CDF term: prefer remainders that are likely to be filled
    cdf_val = _weibull_cdf(r)
    
    # Combine
    # Weight: 0.5 for size, 0.5 for usefulness?
    # Let's try 1.0 for size, 1.0 for CDF
    score = base + 1.0 * cdf_val
    
    scores[mask] = score
    return scores
# EVOLVE-BLOCK-END
