"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import weibull_min

_DEADLINE = 120.0
_start_time = time.time()

# Pre-compute Weibull distribution parameters for typical Weibull sequences
# We assume a standard shape (k) and scale (lambda) that matches the "five generated Weibull sequences"
# Common defaults in bin packing benchmarks are k=1.0 (exponential) or k=2.0 (Rayleigh)
# The prompt mentions "Weibull sequences". Without specific parameters, we assume a general fit.
# However, often these tasks use a specific distribution like Weibull(k=1.0, scale=10) or similar.
# Let's look at the inspirations. Inspiration 1 uses _weibull_cdf.
# A robust approach is to estimate the distribution from the items if possible, but we only see one item at a time.
# We must assume a static distribution. 
# If the items are Weibull, the PDF is f(x) = (k/lambda)(x/lambda)^(k-1)exp(-(x/lambda)^k).
# Let's assume a generic Weibull with k=1.0 (Exponential) or k=2.0.
# Actually, the "L1 lower bound" suggests we are close to optimal.
# A key insight: The "regret" of leaving a gap R is how likely it is that a future item of size <= R arrives.
# If we assume items are i.i.d. from a Weibull distribution, the probability that a future item fits in gap R is CDF(R).
# But we want to maximize the chance of filling the bin.
# A simpler, more effective structural change: 
# Instead of just minimizing gap, we minimize the "wasted potential".
# If we leave a gap R, the expected value of that gap is the probability that it remains unused.
# Let's try a score based on the Weibull CDF of the remaining space.
# If the remaining space R is small, CDF(R) is small -> low probability of being filled -> high regret?
# No, if R is small, it's hard to fill, so it's "waste".
# If R is large, it's likely to be filled by some item.
# So we want to MAXIMIZE CDF(R)? 
# Wait. If I put item i in bin b, remaining = R.
# If R is very small, it's likely to be wasted (no future item fits exactly or close enough).
# If R is large, it's likely to be filled.
# So a "Good" remainder is one that is likely to be filled.
# Probability a single future item fits in R is F(R).
# But we have many future items.
# Let's use the probability that the gap is *not* fillable by the *next* item as a penalty?
# Or better: The expected number of future items that can fit in R is infinite if we wait long enough? No.
#
# Let's look at Inspiration 3: "distribution-aware penalty based on the Weibull CDF to estimate future regret".
# Regret = Expected Waste.
# If we leave gap R, the waste is R if it's never filled.
# Probability it's never filled is low if R is large.
#
# Let's try a different structural approach: **Lookahead with Distribution Matching**.
# We assume the items follow a Weibull distribution. We can estimate the shape and scale from the current item and the bin capacity? No.
# We can't see the sequence.
#
# However, we can use the fact that the items are Weibull. 
# If we assume the items are Weibull(k, lambda), we can define the score of a bin with remaining R as:
# Score = - (Expected Waste)
# Expected Waste approx = R * (1 - F(R))^N ? No.
#
# Let's try a simpler structural change: **Worst-Fit Decreasing logic adapted for online**.
# Worst-Fit places item in the bin with the MOST remaining space.
# This is known to be good for some distributions because it preserves tight bins for small items.
# The current program uses Best-Fit (minimize gap).
# Let's try a hybrid: **Best-Fit Decreasing** is offline. Online, we can use **Harmonic** or **First-Fit Decreasing** logic.
#
# Actually, the prompt says "different strategy... not refining".
# The current program is a weighted Best-Fit.
# Let's try **Worst-Fit** (maximize remaining space) with a cap to avoid wasting huge bins if a tight fit is available?
# Or **First-Fit** (first feasible bin)? No, we get all bins.
#
# Let's try **Worst-Fit**.
# Strategy: Place item in the bin with the largest remaining capacity.
# Why? It keeps "large" bins large for large items, and "small" bins small for small items.
# This is the dual of Best-Fit.
#
# Let's refine Worst-Fit with a distribution-aware twist.
# If we assume Weibull items, small items are more frequent (for k>1) or exponentially distributed.
# For Weibull(k=1), exponential, small items are frequent.
# Worst-Fit works well when small items are frequent because it fills the "holes" in large bins with small items, leaving the large bins intact for other large items.
#
# Let's implement a pure Worst-Fit score:
# Score = Remaining Capacity.
#
# But wait, if I have a bin with 99 remaining and one with 1 remaining, and item is 1.
# Worst-Fit puts it in 99, leaving 98.
# Best-Fit puts it in 1, leaving 0.
# Best-Fit is usually better for packing density.
#
# Let's look at the "Inspiration 1" which scored 1.0309. It used `_weibull_cdf`.
# Let's try to implement a **Regret Minimization** using the Weibull CDF.
# We assume a Weibull distribution. We need parameters.
# If the task generates "five generated Weibull sequences", they might use a standard one.
# Common standard: Weibull(k=1.0, scale=10) or similar.
# If we don't know the parameters, we can't use the CDF directly.
#
# However, we can use a **relative** measure.
# Let's try a **Gap-Squared** penalty but inverted? No.
#
# Let's try **First-Fit Decreasing (FFD) emulation**.
# In FFD, we sort items descending. We can't do that.
#
# Let's try **Next-Fit**? No.
#
# Let's go back to the **Weibull CDF** idea from Inspiration 3.
# We need to guess the Weibull parameters.
# If the items are in [0, 100], a typical Weibull might have scale ~ 20-30.
# Let's assume the items are Weibull(k=1.0) (Exponential) truncated or scaled.
# If we assume Exponential, CDF(x) = 1 - exp(-x/lambda).
#
# Let's try a strategy that **penalizes creating remainders that are "unfit" for the expected distribution**.
# A remainder R is "good" if there is a high probability that a future item of size <= R will arrive.
# Probability that a Weibull item is <= R is F(R).
# We want to maximize F(R).
# So Score = F(R).
#
# But we don't know the Weibull parameters.
# We can estimate them? No, we only see one item.
#
# Let's assume the distribution is **Uniform**? No, it says Weibull.
#
# Let's try a **Hybrid of Best-Fit and Worst-Fit** based on the item size relative to the bin.
# If item is large, use Best-Fit (tight pack).
# If item is small, use Worst-Fit (spread out to preserve tight bins for other smalls? No, smalls fill gaps).
#
# Actually, for Weibull (which has a mode > 0 for k>1), there is a "typical" item size.
#
# Let's try a **Simple Worst-Fit** with a safety check.
# Score = Remaining.
# This is a fundamental structural change from Best-Fit (Score = -Remaining).
#
# Let's try **Worst-Fit**.
#
# Wait, Inspiration 1 scored 1.03. It used `_weibull_cdf`.
# Let's try to replicate that logic more robustly.
# We will assume a Weibull distribution with shape k=1.0 (Exponential) and scale estimated from the item? No.
# Let's assume a fixed Weibull(k=1.0, scale=100/3) ?
#
# Let's try a **Distribution-Aware Best-Fit**.
# Score = -Remaining * (1 - CDF(Remaining))?
#
# Let's try the **Worst-Fit** approach as it is the most distinct structural change from the current Best-Fit-like program.
#
# Revised Plan:
# 1. Check feasibility.
# 2. Score feasible bins by their Remaining Capacity (Worst-Fit).
# 3. Why? Worst-Fit is known to perform well on random instances, especially when items are small relative to bins, by keeping bins "open" for large items.
#
# Let's add a small Best-Fit component to avoid pathological cases?
# No, the directive says "MATERIALLY DIFFERENT".
#
# Let's try **Worst-Fit**.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        return -(bins - item)
    
    remaining = bins - item
    feasible = remaining >= 0
    
    # Worst-Fit: Maximize remaining space
    # This preserves bins with large remaining capacity for future large items.
    # It tends to fill bins with a mix of items, potentially reducing the number of partially filled bins.
    scores = np.where(feasible, remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
