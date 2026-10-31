"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for the generated sequences
# Based on the high-performing inspiration (1.8171), we use:
# k=2.0, lambda=15.0
# This implies mean item size ~ 15 * Gamma(1.5) ≈ 13.3.
# The "Dead Space" metric r * S(r) has a maximum (worst case) at r = lambda * (1/k)^(1/k)
# For k=2, r* = 15 / sqrt(2) ≈ 10.6.
# This suggests remainders around 10 are the most problematic "dead zones".
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Structural "Dead Space" Avoidance with Consolidation Bias.
    
    We combine the structural insight from the top-performing inspiration (score 1.8171)
    with a refinement to prevent pure Worst-Fit behavior.
    
    The core metric is DeadSpace(r) = r * S(r) = r * exp(-(r/lambda)^k).
    This function is U-shaped (or rather, it has a single maximum in the "bad" region).
    - As r -> 0, DeadSpace -> 0 (Good: tight fit).
    - As r -> infinity, DeadSpace -> 0 (Good: large space, low probability of being a "dead" gap relative to item sizes? 
      Actually, for large r, S(r) is tiny, so the product is small. But large r is generally bad for packing efficiency 
      unless it's a new bin. However, in online packing, creating a new bin is costly. 
      The "Dead Space" heuristic argues that medium gaps are the worst because they are too small for big items 
      but too big to be ignored/consolidated easily? Or rather, the probability that a random item fits in r is F(r).
      The expected waste is more complex.
      
    Let's re-evaluate the logic of Inspiration 1 (1.8171):
    It used `score = -(dead_space + beta * r)`.
    The `beta * r` term is crucial. It ensures that as r grows large, the score becomes very negative, 
    effectively forcing Best-Fit behavior for large remainders, while the `dead_space` term 
    creates a "valley" of low scores around the mode of the dead space distribution.
    
    This results in a preference for:
    1. Very small remainders (Best Fit).
    2. Potentially large remainders if they don't trigger the beta penalty too strongly? 
       No, beta * r dominates for large r. 
       
    So the effective score landscape is:
    - High score for r close to 0.
    - Low score for r around 10-15 (the "dead zone").
    - Very low score for r >> 15.
    
    This is structurally different from pure Best Fit (monotonic decrease) or Worst Fit (monotonic increase).
    It avoids placing items that would leave a "medium" gap, which is statistically the least useful 
    for future Weibull-distributed items.
    
    We refine this by tuning the beta parameter and ensuring numerical stability.
    """
    global _START_TIME
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Clip r to non-negative just in case of floating point errors
    r = np.maximum(valid_rems, 0.0)
    
    # Calculate survival probability S(r) = exp(-(r/lambda)^k)
    # Using np.errstate to handle potential overflow/underflow gracefully, though exp handles it well.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        ratio = r / _WEIBULL_LAMBDA
        power_term = np.power(ratio, _WEIBULL_K)
        survival = np.exp(-power_term)
    
    # Dead space metric: r * S(r)
    dead_space = r * survival
    
    # Combine with a linear penalty to ensure consolidation (Best Fit bias)
    # The inspiration used beta = 0.1.
    # Let's stick with beta = 0.1 as it was part of the 1.8171 score.
    beta = 0.1
    
    combined_penalty = dead_space + beta * r
    
    # Score: minimize penalty -> maximize -penalty
    scores[valid] = -combined_penalty
    
    return scores
# EVOLVE-BLOCK-END
