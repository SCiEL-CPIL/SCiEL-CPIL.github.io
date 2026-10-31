import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Valid bins have remaining >= 0
    if item > 50:
        # Large item: Use Worst-Fit to maximize remaining space for small items
        scores = remaining
    else:
        # Small item: Use Best-Fit to minimize remaining space
        scores = -remaining
    
    # Ensure invalid bins are not selected
    # If the evaluator doesn't mask, we should set invalid scores to -inf
    # But the seed didn't, and it worked. 
    # However, to be safe and correct:
    scores = np.where(bins >= item, scores, -np.inf)
    
    return scores
