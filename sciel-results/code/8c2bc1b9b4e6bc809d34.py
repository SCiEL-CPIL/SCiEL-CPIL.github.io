"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    Strategy: Hybrid Best-Fit with Relative Fit adjustment.
    - Primary goal: Minimize remaining space (Best Fit) to consolidate items.
    - Secondary goal: Prefer bins where the item takes a larger fraction of the remaining capacity.
    
    The score is designed to be higher for bins that fit the item tightly.
    We use a combination of absolute gap minimization and relative fit maximization.
    
    score = - (bins - item) + alpha * (item / bins)
    
    This is equivalent to:
    score = -bins + item + alpha * (item / bins)
    score = item + (alpha - 1) * bins + alpha * item / bins
    
    For small alpha, it behaves like Best Fit.
    The term alpha * (item / bins) adds a preference for bins where the item is a significant part of the remaining space, which can help in cases where absolute gaps are similar but relative waste differs.
    
    We also add a very small index-based tie-breaker to ensure deterministic behavior.
    """
    # Ensure bins is a numpy array of floats
    bins = np.asarray(bins, dtype=float)
    
    # Calculate remaining capacity
    remaining = bins - item
    
    # Initialize scores with -inf to exclude bins that cannot fit the item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Mask for valid bins (remaining >= 0 and bins > 0 to avoid division by zero)
    valid = (remaining >= 0) & (bins > 0)
    
    if np.any(valid):
        # Primary component: Best Fit (minimize remaining)
        # Score is higher for smaller remaining
        best_fit_score = -remaining[valid]
        
        # Secondary component: Relative Fit (maximize item/bins)
        # This helps distinguish between bins with similar absolute gaps but different relative waste
        # We need to avoid division by zero, but valid mask ensures bins > 0
        relative_fit_score = item / bins[valid]
        
        # Combine scores
        # alpha controls the weight of the relative fit component
        # A small alpha (e.g., 0.1) keeps the primary Best Fit behavior
        alpha = 0.1
        scores[valid] = best_fit_score + alpha * relative_fit_score
        
        # Add a tiny deterministic tie-breaker to avoid arbitrary selection in ties
        # This helps with reproducibility and consistent behavior
        scores += np.arange(len(bins), dtype=float) * 1e-10
    
    return scores
# EVOLVE-BLOCK-END
