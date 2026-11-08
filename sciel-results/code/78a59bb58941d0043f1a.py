import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    rem = remaining[mask]
    
    # Hybrid weight based on item size
    # Transition from Best-Fit (small items) to Worst-Fit (large items)
    # Capacity is 100.
    # Threshold start: 5.0, Threshold end: 25.0
    w = (item - 5.0) / 20.0
    w = np.clip(w, 0.0, 1.0)
    
    # Score = rem * (2*w - 1)
    # If w=0 (small), score = -rem (Best-Fit)
    # If w=1 (large), score = rem (Worst-Fit)
    scores[mask] = rem * (2.0 * w - 1.0)
    
    return scores
