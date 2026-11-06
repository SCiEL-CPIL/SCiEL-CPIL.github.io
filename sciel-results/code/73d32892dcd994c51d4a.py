import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -np.inf, dtype=float)
    if np.any(valid):
        valid_bins = bins[valid]
        # Hybrid: Best Fit for small items, Worst Fit for large items
        # Use the largest valid bin as a reference for "large"
        max_bin = np.max(valid_bins)
        if max_bin == 0:
            # Edge case, shouldn't happen if valid
            scores[valid] = 0
            return scores
        
        # Normalize item size relative to the largest available space
        # If item is small, weight is low -> Best Fit (-bins)
        # If item is large, weight is high -> Worst Fit (+bins)
        # We use a threshold or a continuous mapping.
        # Let's use a continuous mapping: weight = item / max_bin
        # Score = (2 * weight - 1) * bins
        # This linearly interpolates between -bins and +bins.
        
        weight = item / max_bin
        # Clamp weight to [0, 1] just in case, though item <= max_bin for valid bins
        weight = np.clip(weight, 0, 1)
        
        scores[valid] = (2 * weight - 1) * valid_bins
        
    return scores
