import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Valid bins
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    valid_indices = np.where(valid)[0]
    remaining = bins[valid_indices]
    
    # Best-Fit with a mild preference for bins that are "medium" filled,
    # to avoid both over-packing (which can waste capacity) and under-packing.
    # We use a quadratic penalty on remaining capacity to strongly prefer smaller remainders,
    # but add a small linear term to break ties in favor of slightly larger remainders
    # when the remaining capacities are very close, which can help in some edge cases.
    
    # score = -remaining**2 + 0.01 * remaining
    # The -remaining**2 term dominates, ensuring best-fit behavior.
    # The +0.01 * remaining term provides a small tie-breaker.
    
    scores[valid_indices] = -remaining**2 + 0.01 * remaining
    
    return scores
