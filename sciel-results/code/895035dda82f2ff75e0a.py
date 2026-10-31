import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Only consider bins with enough capacity
    # The problem statement says: "scores each bin with enough remaining capacity"
    # The evaluator likely filters or expects -inf for invalid bins?
    # "each arriving item goes to the highest-scoring bin"
    # If a bin doesn't have enough capacity, it shouldn't be chosen.
    # We should set score to -np.inf for bins that can't fit the item.
    
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # For valid bins, use Reciprocal Best Fit
    # score = 1 / (remaining + epsilon)
    # To avoid division by zero, add epsilon
    eps = 1e-9
    scores[valid] = 1.0 / (remaining[valid] + eps)
    
    return scores
