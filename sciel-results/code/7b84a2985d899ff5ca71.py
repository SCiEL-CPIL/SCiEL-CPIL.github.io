"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences". 
# Common Weibull parameters for bin packing tasks (capacity 100) often involve
# shapes that produce a mix of small and large items.
# Assuming a standard Weibull distribution with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and a median of ~19, which fits well into capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
import scipy.special

k = 2.0
lambda_scale = 25.0
expected_next_item = lambda_scale * scipy.special.gamma(1 + 1/k)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity for the current item
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Multi-dimensional knapsack / Probabilistic Effective Capacity
    # We score each bin based on the "effective capacity" remaining after 
    # accounting for the expected size of the next item.
    # Effective Capacity = bins - item - expected_next_item
    # We want to maximize the remaining space for future items, but not 
    # waste too much space on the current item (Best Fit logic).
    # However, the directive suggests: "prioritizing bins that maximize the 
    # probability of accepting a second consecutive item."
    
    # Let's refine the score:
    # We prefer bins where (bins - item) is large enough to likely fit the next item.
    # Score = - (bins - item) + bonus_if_likely_to_fit_next
    # Or simpler: Score = - (bins - item - expected_next_item)
    # This is equivalent to: Score = -bins + item + expected_next_item
    # Since item and expected_next_item are constant for all bins, this reduces to:
    # Score = -bins
    # Which is Worst Fit. This is likely too aggressive.
    
    # Let's combine Best Fit (minimize gap) with the probabilistic constraint.
    # We want to minimize the gap, but penalize gaps that are too small to 
    # likely accommodate the next item.
    
    gap = bins - item
    # If the gap is less than the expected next item, it's "risky" or "fragmented".
    # We want to avoid creating fragments that are too small.
    # Score = -gap + penalty(gap < expected_next_item)
    
    # A smooth penalty: if gap < expected_next_item, add a quadratic penalty.
    # Or simply: prioritize bins where gap >= expected_next_item, and among those, 
    # pick the one with the smallest gap (Best Fit).
    # If no bin has gap >= expected_next_item, pick the one with the smallest gap (Best Fit)
    # to preserve larger gaps for larger items.
    
    # Let's try: Score = -gap * (1 + exp(-(gap - expected_next_item)/sigma))
    # This smoothly transitions from Best Fit to "Avoid Small Gaps".
    
    sigma = 5.0
    # To avoid overflow in exp, clip the argument
    arg = (gap - expected_next_item) / sigma
    arg = np.clip(arg, -10, 10)
    multiplier = 1 + np.exp(-arg)
    
    scores = -gap * multiplier
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
