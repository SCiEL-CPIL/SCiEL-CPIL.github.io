import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Valid bins
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    # First Fit Strategy:
    # Prefer the bin with the smallest index that still fits the item.
    # This strategy tends to fill bins sequentially, potentially keeping
    # the total number of open bins lower in some distributions.
    # To achieve this with a "highest score wins" mechanism:
    # Assign a score inversely related to the bin index.
    # score = -index.
    # The bin with the smallest index (0) will have the highest score (0).
    
    valid_indices = np.where(valid)[0]
    scores[valid_indices] = -valid_indices
    
    return scores
