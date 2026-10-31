"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters for the generated sequences
# Typical online bin packing benchmarks use Weibull with shape ~1.5-2.0 and scale such that mean ~ 0.1 * capacity
# Let's assume capacity C=100. If mean item size is around 10, scale lambda = 10 / gamma(1+1/k).
# For k=2, gamma(1.5) = 0.886, so lambda = 10/0.886 = 11.28.
# Let's use k=1.5, gamma(1+1/1.5) = gamma(1.666) = 0.9027. lambda = 10/0.9027 = 11.07.
# Let's pick k=1.5, lambda=11.0.

WEIBULL_K = 1.5
WEIBULL_LAMBDA = 11.0
CAPACITY = 100.0
N_ITEMS = 1000
N_SEEDS = 5

# Pre-compute expected future demand statistics or a "waste" metric
# The L1 lower bound is sum(item_sizes) / capacity.
# The goal is to minimize the number of bins used.
# Best-fit decreases the remaining space in the bin.
# A structural change: Instead of just minimizing remaining space (which is local),
# we can estimate how much of the remaining space will actually be used by future items.
# If we assume future items are i.i.d. from the Weibull distribution, we can compute
# the probability that a future item fits in the remaining space r.
# P(item <= r) = 1 - exp(-(r/lambda)^k)
# However, we want to maximize the utilization of the remaining space.
# A better metric might be the expected number of items that can fit in the remaining space?
# Or simply the probability that the space is "wasted" (no future item fits).
# If no future item fits, the space is dead.
# We want to minimize dead space.
# Dead space probability for remainder r: P(item > r) = exp(-(r/lambda)^k).
# So we want to minimize exp(-(r/lambda)^k).
# This is equivalent to maximizing (r/lambda)^k, or just maximizing r? No, that's worst-fit.
# Wait. If r is small, P(item > r) is small (likely to be used? No, if r is small, few items fit).
# Let's re-evaluate.
# If r is very large, many items fit. P(item > r) is small. The space is likely to be used by many items.
# If r is very small, few items fit. P(item > r) is large. The space is likely to be wasted (dead).
# So we want to minimize the probability that the space is wasted?
# Wasted means no future item fits? Or the space remains unused?
# Actually, in online bin packing, once an item is placed, the remaining space is available for future items.
# If we leave a large remainder, it's more likely to be filled by a future item.
# If we leave a small remainder, it's less likely to be filled, so it becomes dead space sooner.
# Therefore, we should PREFER bins with LARGER remainders? That sounds like First-Fit or Worst-Fit.
# But Best-Fit usually works well. Why?
# Because Best-Fit fills bins tightly, reducing the total number of "open" bins with large remainders?
# Actually, Best-Fit minimizes the maximum remainder, which tends to keep the distribution of remainders tight.
#
# Let's look at the "Expected Dead Space" insight from the prompt:
# "Replaces heuristic 'dead zone' estimation with a distribution-aware 'Expected Dead Space' metric (r * S(r))"
# S(r) is the survival function: P(item > r) = exp(-(r/lambda)^k).
# Dead space is the space that will NOT be used by future items.
# If we assume that if an item fits, it will eventually be placed there (greedy),
# the expected dead space is roughly the remainder r multiplied by the probability that no future item fits?
# Or is it the integral of the survival function from r to infinity?
# The expected dead space in a bin with remainder r, assuming items arrive until none fit,
# is not just r * S(r).
#
# Let's try a different structural approach:
# Simulate a short horizon or use a pre-computed lookup table for the expected number of bins
# needed to pack a sequence of future items into a bin with remainder r.
#
# Given the time constraint (150s), we can pre-compute a value function V(r)
# which represents the expected "cost" or "waste" associated with a remainder r.
# The cost is the expected number of bins that will be "wasted" or the expected dead space.
#
# Let's define the score as the negative of the expected dead space.
# We want to minimize dead space.
#
# Pre-computation:
# For a given remainder r, what is the expected dead space?
# This depends on the future item sizes.
# Let's approximate the expected dead space as:
# E[Dead] = r * P(no future item fits) + E[Dead | some future item fits]
# This is complex.
#
# Simpler structural change:
# Use a pre-computed table of "attractiveness" for each possible remainder.
# The attractiveness is based on the probability that the remainder will be fully utilized.
#
# Let's try the metric from the prompt: r * S(r) is NOT the dead space, it's the "risk" of dead space?
# The prompt says: "Expected Dead Space metric (r * S(r))".
# Let's assume this metric is what we want to MINIMIZE.
# So score = - (r * S(r)).
#
# Let's test this logic.
# If r is small, S(r) is close to 1. r * S(r) is small.
# If r is large, S(r) is close to 0. r * S(r) is small.
# There is a maximum in between.
# We want to MINIMIZE r * S(r).
# This means we prefer either very small remainders (Best-Fit) or very large remainders (Worst-Fit).
# Best-Fit is known to be good.
#
# Let's implement:
# 1. Pre-compute the survival function S(r) for r in [0, CAPACITY].
# 2. In priority(), compute r = bins - item.
# 3. Compute metric = r * S(r).
# 4. Return -metric as the score (higher is better, so we minimize metric).

def _compute_survival_table():
    """Pre-compute the survival function S(r) = P(item > r) for r in [0, CAPACITY]."""
    r_values = np.linspace(0, CAPACITY, 1001)
    # S(r) = exp(-(r/lambda)^k)
    s_values = np.exp(-((r_values / WEIBULL_LAMBDA) ** WEIBULL_K))
    return r_values, s_values

# Pre-compute the table
R_TABLE, S_TABLE = _compute_survival_table()

def _interpolate_survival(r):
    """Interpolate the survival function S(r) for a given remainder r."""
    if r <= 0:
        return 1.0
    if r >= CAPACITY:
        return 0.0
    # Find the index in the table
    idx = int(r / CAPACITY * (len(R_TABLE) - 1))
    # Clamp index
    idx = np.clip(idx, 0, len(R_TABLE) - 2)
    # Linear interpolation
    r0 = R_TABLE[idx]
    r1 = R_TABLE[idx + 1]
    s0 = S_TABLE[idx]
    s1 = S_TABLE[idx + 1]
    if r1 == r0:
        return s0
    frac = (r - r0) / (r1 - r0)
    return s0 + frac * (s1 - s0)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    Score is based on minimizing the expected dead space metric r * S(r).
    """
    # Compute remaining capacity for each bin
    remaining = bins - item
    
    # Filter out bins with insufficient capacity
    valid_mask = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # Compute S(r) for each valid remainder
    # Vectorized interpolation
    # r_values is 0 to 100, 1001 points. Step is 0.1.
    idx = np.floor(valid_remaining / CAPACITY * (len(R_TABLE) - 1)).astype(int)
    idx = np.clip(idx, 0, len(R_TABLE) - 2)
    
    r0 = R_TABLE[idx]
    r1 = R_TABLE[idx + 1]
    s0 = S_TABLE[idx]
    s1 = S_TABLE[idx + 1]
    
    # Avoid division by zero
    denom = r1 - r0
    denom[denom == 0] = 1.0
    frac = (valid_remaining - r0) / denom
    s_values = s0 + frac * (s1 - s0)
    
    # Compute metric: r * S(r)
    # We want to minimize this, so score = -metric
    metric = valid_remaining * s_values
    scores[valid_mask] = -metric
    
    return scores
# EVOLVE-BLOCK-END
