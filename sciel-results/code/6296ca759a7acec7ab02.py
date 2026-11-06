# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Uses a First-Fit-Diminishing inspired heuristic:
    - Primary: prefer bins with the least remaining capacity after placing the item
      (i.e., minimize waste in the chosen bin).
    - Secondary: among ties or near-ties, prefer bins that have already been used
      (lower remaining capacity before placement indicates consolidation).
    
    The score returned is such that higher score = better bin.
    We return negative of remaining capacity after placement, so bins that leave
    less space get higher scores. This aggressively consolidates items.
    """
    remaining = bins - item
    # Only bins with enough capacity are valid
    valid = remaining >= 0
    
    # For invalid bins, set score to -inf so they are never chosen
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # For valid bins, score = -(remaining after placement)
        # This means bins that leave less space get higher (less negative) scores
        # Equivalently: prefer bins where (bins - item) is smallest
        scores[valid] = -remaining[valid]
        
        # Add a small tie-breaking bonus for bins that are more "full" already
        # This helps consolidate: among bins with similar remaining capacity after
        # placement, prefer those that were already more utilized
        # The term bins[valid] * epsilon encourages choosing fuller bins
        epsilon = 1e-6
        scores[valid] += epsilon * bins[valid]
    
    return scores
# EVOLVE-BLOCK-END
