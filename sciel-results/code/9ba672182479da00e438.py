import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # First-Fit strategy:
    # Assign a score of 0 to all bins that can accommodate the item.
    # Assign a very low score to bins that cannot.
    # np.argmax will select the first bin with score 0 (lowest index).
    # This preserves later bins for potentially larger items.
    
    remaining = bins
    # Valid bins have remaining >= item
    valid_mask = remaining >= item
    
    scores = np.full(len(bins), -1e18)
    scores[valid_mask] = 0.0
    
    return scores
