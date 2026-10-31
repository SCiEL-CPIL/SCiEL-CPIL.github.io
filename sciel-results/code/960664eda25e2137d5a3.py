import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    The item goes to the highest-scoring bin.
    
    Strategy:
    - For small items: Best-Fit (minimize remaining space) to pack tightly.
    - For large items: Worst-Fit (maximize remaining space) to preserve fragmentation.
    - Hybrid: Use a smooth transition based on item size relative to bin capacity.
    
    We use a piecewise scoring function:
    - If item is small (relative to capacity), prefer bins where it fits tightly (Best-Fit).
    - If item is large, prefer bins with more free space (Worst-Fit).
    
    The L1 lower bound for bin packing is related to the sum of item sizes / capacity.
    To minimize excess over L1, we want to minimize the number of bins used, which means
    packing items as tightly as possible. However, for very large items, worst-fit can
    help avoid wasting space in bins that are already partially filled.
    
    A common effective heuristic is:
    - Best-Fit Decreasing (BFD) when items are sorted, but here items arrive online.
    - For online, a blend of Best-Fit and Worst-Fit works well.
    
    Let's use:
    - score = -(remaining) for small items (Best-Fit)
    - score = remaining for large items (Worst-Fit)
    - Transition smoothly around item = 50 (half capacity)
    """
    remaining = bins - item
    
    # Valid bins must have remaining >= 0
    valid = bins >= item
    
    # Base scores
    # For small items, Best-Fit: minimize remaining -> score = -remaining
    # For large items, Worst-Fit: maximize remaining -> score = remaining
    
    # Use a smooth transition. Let's use a sigmoid-like blend.
    # When item is small, weight toward Best-Fit; when large, toward Worst-Fit.
    
    # Simple approach: threshold at 50
    if item <= 50:
        scores = -remaining
    else:
        scores = remaining
    
    # Mask invalid bins
    scores = np.where(valid, scores, -np.inf)
    
    return scores
