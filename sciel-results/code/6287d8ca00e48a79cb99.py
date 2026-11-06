"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: total wall-clock limit is 150s, but evaluator kills at 120s for scoring.
# We use a conservative internal deadline to ensure we return before being killed.
_DEADLINE = 115.0  # seconds, safety margin
_start_time = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    We use a heuristic that balances best-fit with a penalty for fragmentation.
    
    For online bin packing with Weibull items, a good heuristic is:
    - Prefer bins where the item fits with minimal waste (best-fit)
    - But avoid creating bins that are too "tight" if it leads to fragmentation
    - A common improvement over pure best-fit is to consider the "next-fit decreasing" 
      intuition or use a weighted combination of remaining space and used space.
    
    Strategy: Use a score that favors bins where the remaining space after placing 
    the item is small (best-fit), but with a slight preference for bins that are 
    already partially filled (to avoid opening new bins unnecessarily when possible).
    
    Specifically:
    - For bins with capacity >= item:
      - remaining_after = bin_capacity - item
      - We want to minimize remaining_after (best-fit)
      - But we can add a small bonus for bins that are more "full" before placement
        to encourage consolidation.
    
    Let's use: score = -(remaining_after) + alpha * (used_before / capacity)
    where used_before = capacity - remaining_before, and remaining_before = bin_capacity.
    
    Actually, let's think differently. The key insight from the literature for 
    online bin packing with known capacity is that Best-Fit Decreasing (BFD) is 
    optimal offline, but online we don't know future items. For Weibull distributions,
    items are often skewed. 
    
    A robust approach is to use a "Worst-Fit Decreasing" variant or a hybrid.
    However, for online bin packing, Best-Fit is generally strong. Let's try a 
    refined Best-Fit that considers the ratio of remaining space.
    
    Let's use: score = -(bins - item) / (bins + epsilon)
    This normalizes the remaining space by the bin capacity, which can help when
    bins have different capacities (though here they are all 100).
    
    Since all bins have the same capacity (100), bins - item is just the remaining space.
    So -(bins - item) is just -remaining_space, which is Best-Fit.
    
    To improve, let's try a heuristic that considers both the immediate fit and 
    a penalty for bins that are very empty (to encourage filling bins).
    
    score = -(bins - item) - beta * max(0, threshold - bins)
    where threshold is some fraction of capacity.
    
    Alternatively, let's use a simple but effective heuristic: 
    Prefer bins where the item fits, and among those, prefer the one with the 
    smallest remaining space after placement. This is Best-Fit.
    
    To break ties or improve, we can add a small random component or use a 
    more sophisticated score. However, for deterministic evaluation, let's 
    stick with a deterministic improvement.
    
    One known improvement is to use "First-Fit Decreasing" when items are sorted,
    but we don't have that luxury online. 
    
    Let's try: score = -(bins - item) + 0.1 * bins
    This adds a small preference for larger bins (which are all the same, so no effect).
    
    Since all bins have the same capacity, the only difference is how much is used.
    Let's define:
    - remaining = bins - item
    - used = 100 - bins (since capacity is 100)
    
    We want to minimize remaining, but also avoid creating bins with very little 
    remaining space that might not be useful for future items.
    
    A good heuristic from the literature is to use a "score" that is the 
    remaining space, but with a twist: prefer bins where the remaining space 
    is "compatible" with likely future item sizes. Since we don't know the 
    distribution, we can use a heuristic based on the median or mean item size.
    
    However, for simplicity and robustness, let's use a refined Best-Fit:
    score = -(bins - item)
    
    But wait, the seed program already does this. We need to improve it.
    
    Let's try a "Worst-Fit" variant: prefer bins with the most remaining space 
    after placement. This can help in some cases by keeping bins more flexible.
    
    Or a hybrid: score = -(bins - item) + gamma * bins
    where gamma is a small positive number.
    
    Let's experiment with a few heuristics and pick the one that seems most promising.
    
    Given the constraints, let's use a time-aware approach: if we have time, 
    we can try a few different heuristics and pick the best one based on a 
    quick simulation. But that might be too complex.
    
    Instead, let's use a well-known heuristic: "Next-Fit Decreasing" is not 
    applicable online. "Best-Fit Decreasing" is offline. For online, 
    "Best-Fit" is a strong baseline.
    
    To improve Best-Fit, we can use a "lookahead" heuristic: estimate the 
    probability that a bin with remaining space R can accommodate future items.
    This is complex, so let's use a simpler approximation.
    
    Let's try: score = -(bins - item) + 0.05 * (100 - bins)
    This adds a small bonus for bins that are already more full.
    
    Let's go with this.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback to simple Best-Fit if we're running out of time
        remaining = bins - item
        scores = np.full_like(bins, -np.inf, dtype=np.float64)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores
    
    remaining = bins - item
    valid = remaining >= 0
    
    # Compute a refined score
    # Primary: minimize remaining space after placement (Best-Fit)
    # Secondary: prefer bins that are already more full (to reduce fragmentation)
    # We use a weighted combination:
    # score = -remaining + alpha * used
    # where used = 100 - bins (assuming capacity is 100)
    
    # However, we should not assume capacity is 100. Let's infer it from the max bin value.
    # But the task says capacity is 100, so we can hardcode it for speed.
    capacity = 100.0
    used = capacity - bins
    
    # Weighted score:
    # We want to minimize remaining, so -remaining is good.
    # We also want to prefer bins that are more full, so +used is good.
    # Let's use a small alpha to not dominate the Best-Fit behavior.
    alpha = 0.1
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        scores[valid] = -remaining[valid] + alpha * used[valid]
    
    return scores
# EVOLVE-BLOCK-END
