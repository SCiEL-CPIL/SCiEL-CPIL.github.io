import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Hybrid Strategy:
    # For large items (> C/2), use Best Fit to preserve larger bins for other large items.
    # For small items (<= C/2), use First Fit to consolidate items into earlier bins,
    # potentially closing bins sooner and reducing the number of open bins.
    
    if item > 50.0:
        # Best Fit: Minimize remaining space
        rem = bins[feasible] - item
        scores[feasible] = -rem
    else:
        # First Fit: Pick the first feasible bin (lowest index)
        # Score is higher for lower indices
        indices = np.arange(n_bins)
        scores[feasible] = -indices[feasible]
        
    return scores
