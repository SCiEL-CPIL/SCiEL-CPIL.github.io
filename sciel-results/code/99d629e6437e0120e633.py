"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses a refined best-fit heuristic with:
    1. Primary: minimize remaining capacity (best-fit)
    2. Secondary: penalize bins with large remaining capacity quadratically
    3. Tertiary: use a smooth surrogate for smoother optimization landscape
    
    This is designed to improve upon pure best-fit by considering the
    overall distribution of remaining capacities across bins.
    """
    # Track remaining capacity for each bin
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # If only one valid bin, return a neutral score
    if n_valid == 1:
        scores[valid_mask] = 0.0
        return scores
    
    # Strategy: Use a parameterized scoring function with smooth surrogate
    # We want to minimize the final number of bins used.
    # L1 lower bound is sum(items) / capacity.
    # Excess = (bins_used - L1_bound) / L1_bound * 100
    
    # The key insight: we want to minimize the "waste" in terms of 
    # how much capacity is left unused relative to what's needed.
    
    # Use a convex combination with smooth surrogate:
    # Primary: minimize remaining capacity (best-fit)
    # Secondary: prefer bins where the remaining capacity is not too large
    #   (to avoid creating bins with lots of unused space)
    
    # Score = -remaining - lambda * remaining^2 / capacity
    # The quadratic term penalizes bins with large remaining capacity
    
    # Normalize remaining capacities for better scaling
    max_rem = np.max(valid_remaining)
    if max_rem <= 0:
        scores[valid_mask] = 0.0
        return scores
    
    # Use a smooth min surrogate: soft-min of remaining capacities
    # This gives a smoother landscape than hard min
    temperature = 5.0  # Lower = closer to hard min
    
    # Alternative: use a weighted combination that emphasizes tight fits
    # Score = -(remaining + lambda * remaining^2 / capacity + mu * remaining^3 / capacity^2)
    
    # Adaptive lambda based on how many bins are valid
    # When few bins are valid, be more aggressive with best-fit
    # When many bins are valid, be more conservative to balance load
    lambda_param = 0.05 + 0.1 * (n_valid / max(n_valid, 1))
    mu_param = 0.01
    
    # Compute scores with cubic term for stronger penalty on large remaining
    valid_scores = -(valid_remaining + lambda_param * (valid_remaining ** 2) / _capacity 
                     + mu_param * (valid_remaining ** 3) / (_capacity ** 2))
    
    # Add a small deterministic tie-breaker based on bin index to avoid randomness
    # This ensures reproducibility and consistent behavior
    valid_indices = np.where(valid_mask)[0]
    tie_breaker = np.linspace(0, 0.005, n_valid)
    valid_scores += tie_breaker
    
    scores[valid_mask] = valid_scores
    
    return scores


# Pre-compute some parameters that might be useful
# The capacity is given as 100 in the task description
_CAPACITY = 100.0
_LAMBDA_BASE = 0.05
_LAMBDA_SCALE = 0.1
_MU = 0.01
_TEMPERATURE = 5.0

# EVOLVE-BLOCK-END
