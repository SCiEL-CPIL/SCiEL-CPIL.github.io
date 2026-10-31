import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Hybrid Best-Fit / Worst-Fit strategy:
    # We want to balance tight packing (Best-Fit) with preserving space for large items (Worst-Fit).
    # For small items, we prefer tight fits (Best-Fit).
    # For large items, we prefer bins with more remaining space (Worst-Fit) to avoid fragmentation.
    
    # Normalize remaining capacity to [0, 1]
    max_remaining = np.max(remaining[valid]) if np.any(valid) else 1.0
    if max_remaining <= 0:
        max_remaining = 1.0
    
    normalized_remaining = remaining / max_remaining
    
    # Item size relative to typical item size (assume capacity 100, so typical item ~ 10-20)
    # For small items (< 10), use Best-Fit (score = -remaining)
    # For large items (> 30), use Worst-Fit (score = remaining)
    # For medium items, blend the two
    
    # Blend factor: 0 for Best-Fit, 1 for Worst-Fit
    # Small items: blend = 0 (Best-Fit)
    # Large items: blend = 1 (Worst-Fit)
    blend = np.clip((item - 10) / 20, 0, 1)
    
    # Score = blend * remaining + (1 - blend) * (-remaining)
    # = remaining * (2*blend - 1)
    # When blend = 0: score = -remaining (Best-Fit)
    # When blend = 1: score = remaining (Worst-Fit)
    
    score_factor = 2 * blend - 1
    scores = np.where(valid, remaining * score_factor, -np.inf)
    
    return scores
