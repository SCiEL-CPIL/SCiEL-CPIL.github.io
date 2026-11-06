"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track bin usage statistics across calls
_start_time = None
_CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses Pure Best-Fit heuristic:
    Score = -remaining capacity.
    This selects the bin with the least remaining space (tightest fit).
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        remaining = bins - item
        valid_mask = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores
    
    remaining = bins - item
    valid_mask = remaining >= 0
    
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # Pure Best-Fit: Minimize remaining capacity
    # Score is negative remaining capacity
    valid_scores = -valid_remaining
    
    # Add a tiny tie-breaker to ensure deterministic behavior if remaining capacities are identical
    # We prefer the bin with the lower index (which corresponds to the first encountered in the array)
    # but since we are assigning to a subset, we can just add a tiny gradient based on index in the valid set
    n_valid = len(valid_remaining)
    if n_valid > 1:
        # Tie-breaker: prefer earlier bins in the original array?
        # The valid_mask indices are not directly available here as a simple range if we just sliced.
        # We can reconstruct indices.
        valid_indices = np.where(valid_mask)[0]
        # Map valid_indices to 0..n_valid-1 for tie-breaking
        # Actually, just using a linspace on the scores is enough to break ties arbitrarily but deterministically
        tie_breaker = np.linspace(0, 1e-9, n_valid)
        valid_scores += tie_breaker
        
    scores[valid_mask] = valid_scores
    
    return scores

# EVOLVE-BLOCK-END
