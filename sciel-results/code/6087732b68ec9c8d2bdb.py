"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adaptive Best-Fit with smooth sigmoid transition.
    - For large items (relative to bin capacity), strongly prefer Best-Fit (minimize remaining).
    - For small items, be more neutral to avoid fragmentation (Worst-Fit tendency for small items).
    - This is achieved via a smooth sigmoid that blends between Best-Fit and a neutral score.
    """
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores with a very low value for infeasible bins
    scores = np.full(len(bins), -1e18)
    
    if not np.any(feasible):
        return scores
    
    # For feasible bins, compute adaptive score
    # Best-Fit score: -remaining (prefer bins with smallest remaining capacity)
    # Neutral score: 0 (no preference)
    # Blend factor: sigmoid that is ~1 for large items, ~0 for small items
    
    # item_ratio = item / max(bins) to normalize item size
    max_bin = np.max(bins)
    if max_bin > 0:
        item_ratio = item / max_bin
    else:
        item_ratio = 0.0
    
    # Sigmoid transition: for item_ratio > 0.5, strongly prefer Best-Fit
    # For item_ratio < 0.3, more neutral
    # Using a smooth sigmoid: weight = 1 / (1 + exp(-k * (item_ratio - threshold)))
    # threshold = 0.4, k = 10
    weight = 1.0 / (1.0 + np.exp(-10.0 * (item_ratio - 0.4)))
    
    # Blend: score = weight * (-remaining) + (1 - weight) * 0
    # This smoothly transitions from Best-Fit to neutral as item size decreases
    scores[feasible] = weight * (-remaining[feasible])
    
    return scores
# EVOLVE-BLOCK-END
