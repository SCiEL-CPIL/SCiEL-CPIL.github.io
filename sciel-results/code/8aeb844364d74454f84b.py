"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid Best-Fit with size-dependent adaptation.
    - For small items (item < 50): use Best-Fit (minimize remaining after placement)
      to consolidate items into fewer bins.
    - For large items (item >= 50): use Worst-Fit (maximize remaining after placement)
      to keep bins open for other large items, reducing fragmentation.
    
    This addresses the key insight that pure Best-Fit can be suboptimal for large items
    because it may fill bins to a point where no other large item fits, whereas
    Worst-Fit preserves more flexible space.
    
    The threshold of 50 (half of capacity 100) is chosen because:
    - Items >= 50 can only share a bin with items <= 50
    - Items < 50 can share with a wider range of items
    - Best-Fit works well for consolidation of smaller items
    - Worst-Fit works well for preserving flexibility for large items
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    # Determine strategy based on item size
    # Capacity is 100, so items >= 50 are "large"
    if item >= 50.0:
        # Worst-Fit: prefer bins with the MOST remaining capacity after placement
        # This keeps bins open for other large items
        # Score = remaining_after (higher is better for feasible bins)
        scores = np.full(len(bins), -np.inf)
        scores[feasible] = remaining_after[feasible]
    else:
        # Best-Fit: prefer bins with the LEAST remaining capacity after placement
        # This consolidates items into fewer bins
        # Score = -remaining_after (higher is better for feasible bins)
        scores = np.full(len(bins), -np.inf)
        scores[feasible] = -remaining_after[feasible]
    
    # Add a tiny deterministic tie-breaker favoring lower indices
    # This ensures stable behavior when scores are identical
    # The epsilon is small enough not to disturb the primary ranking
    indices = np.arange(len(bins))
    scores = scores - 1e-10 * indices
    
    return scores
# EVOLVE-BLOCK-END
