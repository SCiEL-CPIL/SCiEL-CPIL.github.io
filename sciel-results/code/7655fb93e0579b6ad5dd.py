import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start = time.time()
    deadline = start + 10  # 10 second safety margin
    
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf)
    
    if not np.any(valid):
        return scores
    
    if item > 50:
        scores[valid] = remaining[valid]
    else:
        scores[valid] = -remaining[valid]
    
    return scores
