"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    We want to minimize the number of bins used (or equivalently, the excess over L1 lower bound).
    
    Strategy: Best-Fit Decreasing variant with a twist.
    - First, filter bins that can fit the item.
    - Among those, we want to pick the bin that leaves the least remaining space (Best Fit).
    - However, simply doing Best Fit can be suboptimal. A common improvement is to
      prefer bins that are already "partially used" in a way that complements the item,
      or to use a slight penalty for bins that are almost full to avoid fragmentation.
      
    Actually, for online bin packing, the classic Best-Fit (BF) and Best-Fit Decreasing (BFD) 
    are strong. Since we don't know future items, we can't do BFD. 
    But we can refine Best Fit.
    
    One known heuristic is "First Fit Decreasing" but we don't have decreasing order.
    Another is "Worst Fit" which tries to keep bins empty for large items, but large items 
    are already in the sequence.
    
    Let's try a hybrid: 
    - If the item is large (relative to capacity), Best Fit is usually good.
    - If the item is small, maybe we should pack it into a bin that has a lot of space 
      but not too much? Or maybe Best Fit is still best?
    
    Research suggests that for random inputs, Best Fit performs well. 
    Let's try to add a small randomization or a tie-breaking mechanism.
    But the evaluator runs 5 sequences. We need determinism or consistent behavior.
    
    Let's look at the "excess over L1 lower bound". 
    L1 lower bound is sum(items) / capacity.
    To minimize bins, we want to maximize the fill of each bin.
    Best Fit minimizes the remaining space in the chosen bin.
    
    Let's try "Best Fit" with a tie-breaker that prefers bins with higher current usage?
    Or maybe "Worst Fit" for small items?
    
    Actually, a simple improvement over Best Fit is to consider the "density" or "efficiency".
    
    Let's try: Score = -(bins - item) + epsilon * bins
    This slightly prefers bins that are already fuller, which might help consolidate items.
    Wait, if I prefer bins that are fuller, I might fill them up faster, which is good.
    But if I fill one bin completely, I open a new one.
    Best Fit already does this by minimizing remaining space.
    
    Let's try a different approach: 
    "Next Fit" is bad. "First Fit" is okay. "Best Fit" is good.
    
    Maybe we can use a look-ahead heuristic? No, we don't have future items.
    
    Let's try to penalize bins that are very close to full? No, that's what Best Fit does.
    
    What if we use a "soft" Best Fit? 
    Score = -exp(-(bins - item))
    This makes the difference between "very tight" and "tight" less significant.
    
    Or, let's try "Worst Fit" for items that are small relative to the remaining space?
    
    Let's stick to a refined Best Fit. 
    The current seed is `-(bins - item)`.
    
    One known trick is to add a small random noise to break ties, but we want determinism.
    
    Let's try "Best Fit" with a preference for bins that have a usage ratio close to 0.5?
    No, that's arbitrary.
    
    Let's try "Worst Fit" (maximize remaining space) and see if it helps for small items.
    Actually, for online bin packing with unknown distribution, Best Fit is generally superior 
    to First Fit and Worst Fit for minimizing the number of bins.
    
    However, there is a heuristic called "Best Fit Decreasing" which is optimal for offline. 
    Since we are online, we can't sort.
    
    Let's try to combine Best Fit and Worst Fit.
    If the item is large, use Best Fit. If the item is small, use Worst Fit?
    Threshold: 0.5 * capacity?
    
    Let's try:
    If item > 0.5: Best Fit (minimize remaining)
    Else: Worst Fit (maximize remaining)
    
    Why? Large items need to go into bins where they fit well to avoid waste.
    Small items can go into bins with lots of space to "save" the tight spaces for larger items 
    that might come later. This is a form of "reserving" capacity.
    
    Let's implement this hybrid.
    """
    capacity = 100.0  # From task description
    
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return -np.inf * np.ones_like(bins)
    
    scores = -np.inf * np.ones_like(bins)
    
    # Hybrid strategy
    threshold = 0.5 * capacity
    
    if item > threshold:
        # Best Fit: minimize remaining space
        # Score = -remaining
        scores[valid_mask] = -remaining[valid_mask]
    else:
        # Worst Fit: maximize remaining space
        # Score = remaining
        scores[valid_mask] = remaining[valid_mask]
    
    return scores
# EVOLVE-BLOCK-END
