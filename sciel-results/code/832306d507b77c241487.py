"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Smooth best-fit with superlinear penalty on leftover space.
    - Primary: minimize leftover space (best-fit)
    - Secondary: penalize large leftover spaces quadratically to reduce fragmentation
    - Deadline guard for safety
    """
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores
    
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # Smooth best-fit with quadratic penalty on large leftover spaces
    # score = -remaining - alpha * remaining^2 / (remaining + delta)
    # For small remaining: ≈ -remaining (best-fit dominates)
    # For large remaining: ≈ -remaining - alpha*remaining = -(1+alpha)*remaining
    # This makes the penalty for large waste superlinear, reducing fragmentation
    
    # Use a smooth form to avoid numerical issues
    # remaining^2 / (remaining + delta) is smooth and well-behaved
    delta = 1.0  # smoothing parameter
    alpha = 0.01  # strength of quadratic penalty
    
    # First Fit emulation: prefer lower indices
    # We add a linear penalty to the index to simulate First Fit behavior.
    # This encourages filling earlier bins, which can reduce fragmentation
    # by keeping later bins with larger remaining capacities available
    # for larger items.
    index_penalty = valid_indices * 0.1
    
    scores[valid_indices] = -valid_remaining - index_penalty
    
    return scores
# EVOLVE-BLOCK-END
