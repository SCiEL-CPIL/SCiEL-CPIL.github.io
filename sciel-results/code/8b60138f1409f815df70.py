import numpy as np
import time

# We assume capacity is 100 as per task description
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    valid_rems = remaining[valid_idx]
    
    # Strategy: Hybrid of Rank-based Best Fit (from current) and Next-Item Awareness (from inspirations)
    # 
    # 1. Rank-based Best Fit:
    #    The current program uses a rank-based score: score_i = -count(remaining[j] < remaining[i]).
    #    This favors bins that leave a small gap (Best Fit behavior) but is robust to outliers compared to raw min.
    #    
    # 2. Next-Item Awareness:
    #    We estimate the "typical" next item size. Since we don't have the item stream, we can infer
    #    the scale of items from the current bin occupancies or the distribution of gaps.
    #    However, a simpler and robust proxy for "Next-Item Awareness" in online packing with Weibull
    #    distributions (which are skewed, many small, few large) is to consider the *median* or *mean* 
    #    of the current gaps as a proxy for what size of item might fit in the future.
    #    
    #    Actually, a better "Next-Item" heuristic is to prefer bins where the remaining space is 
    #    "compatible" with the distribution of remaining spaces in other bins. 
    #    If we place an item in bin i, we create a new gap r_i. 
    #    We want r_i to be a size that is likely to be filled by future items.
    #    
    #    Let's refine the current rank-based approach by adding a penalty/bonus based on 
    #    how "isolated" the gap is.
    #    
    #    Alternative Idea from Inspiration 1 & 2: Smooth blend of Best-Fit and Worst-Fit.
    #    Current program is essentially Best-Fit (via rank). 
    #    For large items, Worst-Fit is often better to avoid fragmentation.
    #    
    #    Let's combine:
    #    - Calculate the rank-based best-fit score (current logic).
    #    - Calculate a worst-fit score (max remaining).
    #    - Blend them based on item size.
    #    
    #    However, the directive says "KEEP this design and investigate improvements within its family".
    #    The current design is "Rank-based Best Fit". 
    #    The strongest idea from inspirations is the "Threshold/Blend" logic (Inspiration 1 & 2).
    #    
    #    Let's try to enhance the rank-based score with a "gap utility" term.
    #    The rank-based score is: -rank(remaining[i])
    #    This is a discrete step function. 
    #    
    #    Let's look at Inspiration 1: Threshold 0.5*CAPACITY.
    #    If item > 50, use Worst Fit (max remaining).
    #    Else, use Best Fit (min remaining).
    #    
    #    The current program's rank-based best fit is a sophisticated "min remaining".
    #    Let's apply the threshold logic:
    #    If item is large (> 50), switch to a Worst-Fit-like strategy.
    #    If item is small, keep the Rank-Based Best-Fit.
    #    
    #    Why? Large items fragment the bin. If we put a large item in a bin with just enough space (Best Fit),
    #    the leftover is small and useless. If we put it in a bin with lots of space (Worst Fit),
    #    the leftover is large and might fit another large item.
    #    
    #    So:
    #    if item > 50:
    #         score = remaining (Worst Fit)
    #    else:
    #         score = -rank(remaining) (Current Program Logic)
    
    if item > 0.5 * CAPACITY:
        # Large item: Prefer bins with the most remaining space to avoid creating unusable small gaps.
        # This is the "Worst Fit" strategy.
        scores[valid_idx] = valid_rems
    else:
        # Small/Medium item: Use the current program's rank-based best fit.
        # This minimizes the gap while being robust to the distribution of gaps.
        sorted_rems = np.sort(valid_rems)
        # Count of elements strictly less than r
        counts = np.searchsorted(sorted_rems, valid_rems, side='left')
        # Higher score is better. We want small counts (small gaps).
        # So score = -counts.
        scores[valid_idx] = -counts
        
    return scores
