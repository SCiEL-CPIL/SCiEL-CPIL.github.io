import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    if len(bins) == 0:
        return np.array([])
    
    # Filter bins that can fit the item
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # If no bin fits, return empty (evaluator handles new bin)
    if not np.any(valid_mask):
        return np.array([])
    
    valid_remaining = remaining[valid_mask]
    
    # Best-Fit score: lower remaining is better (more negative is worse score if we maximize)
    # We want to maximize the score.
    # Best-Fit: minimize remaining. So score = -remaining.
    best_fit_scores = -valid_remaining
    
    # Neutral score: 0
    neutral_scores = np.zeros_like(valid_remaining)
    
    # Sigmoid weight
    # item is float, capacity is 100
    # w = 1 / (1 + exp(-k * (item - threshold)))
    # We want w -> 1 for large item, w -> 0 for small item
    threshold = 50.0
    k = 10.0
    w = 1.0 / (1.0 + np.exp(-k * (item - threshold)))
    
    # Blend
    scores = w * best_fit_scores + (1.0 - w) * neutral_scores
    
    # Map back to full array
    full_scores = np.full(len(bins), -np.inf)
    full_scores[valid_mask] = scores
    
    return full_scores
