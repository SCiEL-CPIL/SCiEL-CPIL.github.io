import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0

    # Next-Fit strategy:
    # Assign the item to the first (lowest-index) bin that has enough remaining capacity.
    # To make the highest-scoring bin be the lowest-index valid bin, we score by -index.
    # Lower index -> higher (less negative) score -> selected first.
    # This creates a sequential filling pattern where bins are filled in order,
    # which can be effective for Weibull distributions where item sizes have a
    # particular structure.
    
    indices = np.arange(len(bins), dtype=np.float64)
    scores = np.where(valid, -indices, -np.inf)
    
    return scores
