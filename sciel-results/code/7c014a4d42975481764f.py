"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 150.0  # seconds

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses a harmonic mean based score that penalizes both tiny leftover gaps
    and excessively large remaining spaces. This approximates a
    "best-fit-decreasing" ideal by favoring bins where the item size is
    proportional to the remaining capacity.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Compute harmonic mean of remaining capacity and item size
    # H = 2 * (remaining * item) / (remaining + item)
    # We want to maximize this, so higher harmonic mean is better
    # But we also want to prefer bins with less remaining (best-fit)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        harmonic = np.where(valid, 2 * remaining * item / np.maximum(remaining + item, 1e-9), 0)
    
    # Normalize: we want bins where remaining is close to item to score high
    # But also penalize very large remaining (fragmentation risk)
    
    # Score = harmonic mean - beta * max(0, remaining - k*item)
    # This gives a bonus for good fit and a penalty for excessive space
    
    beta = 0.2
    k = 1.5  # threshold multiple of item size
    
    excess_penalty = np.where(valid, beta * np.maximum(0, remaining - k * item), 0)
    
    # Also add a small bonus for very tight fits (remaining close to item)
    # This helps avoid leaving unusable gaps
    tight_bonus = np.where(valid, 0.1 * np.exp(-np.abs(remaining - item) / np.maximum(item, 1e-9)), 0)
    
    score = harmonic - excess_penalty + tight_bonus
    
    # Mask invalid bins
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
