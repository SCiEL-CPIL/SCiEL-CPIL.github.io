import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin based on a shifted polarization strategy.
    Bins that can be filled to a remainder below 80 are preferred (Best Fit behavior).
    If no such bin exists, bins that stay most empty are preferred (Worst Fit behavior).
    """
    C = 100.0
    
    remainders = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    valid_mask = remainders >= 0
    
    if np.any(valid_mask):
        valid_remainders = remainders[valid_mask]
        
        # Shifted center to 80:
        # - If a bin can be filled to remainder < 80, it gets a high score (Best Fit)
        # - Otherwise, bins with larger remainders get higher scores (Worst Fit)
        center = 80.0
        scores[valid_mask] = (valid_remainders - center) ** 2
    
    return scores
