"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Uses a polarization strategy: prefer bins that result in a remainder
    far from half-capacity (50). This consolidates small items into full bins
    or leaves large bins empty, preserving half-full bins for medium items.
    """
    capacity = 100.0
    half_cap = capacity / 2.0
    
    # Calculate remaining capacity if item is placed
    # We only care about bins where bins >= item
    # For invalid bins, we set a very low score so they are never picked
    
    # Vectorized calculation
    # Note: bins - item can be negative for invalid bins
    remainders = bins - item
    
    # Mask for valid bins
    valid_mask = remainders >= 0
    
    # Calculate score: distance from half capacity
    # We want to maximize this score
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # Only compute for valid bins to avoid issues with negative remainders
    # though abs() would handle it, we want to ensure invalid bins are low
    if np.any(valid_mask):
        valid_remainders = remainders[valid_mask]
        valid_scores = np.abs(valid_remainders - half_cap)
        scores[valid_mask] = valid_scores
        
    return scores
# EVOLVE-BLOCK-END
