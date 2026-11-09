"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the highest-scoring bin.
    
    Strategy: Use a hybrid of Best-Fit and a penalty for creating "wasted" space.
    We prefer bins where the item fits tightly (small residual), but also consider
    that leaving moderate space might be useful for future items.
    
    The score is designed to minimize the expected number of bins used by:
    1. Strongly preferring tight fits (best-fit behavior)
    2. Penalizing residuals that are too large (wasting capacity)
    3. Slight preference for bins that are already more filled (consolidation)
    """
    remaining = bins
    capacity = 100.0
    
    # Only consider bins that can fit the item
    mask = remaining >= item
    if not np.any(mask):
        return np.full_like(remaining, -np.inf)
    
    residual = remaining[mask] - item
    
    # Base score: negative residual (best-fit preference)
    # This strongly prefers bins where the item fits most tightly
    
    # Add a quadratic penalty for large residuals to discourage wasting space
    # The penalty grows with residual^2, making large residuals very costly
    
    # Normalize residual to [0, 1] for consistent penalty scaling
    norm_residual = residual / capacity
    
    # Score: -residual (best-fit) - alpha * residual^2 (waste penalty)
    # alpha controls how strongly we penalize large residuals
    alpha = 0.5
    
    score = -residual - alpha * (residual ** 2)
    
    # Map back to full array
    full_score = np.full_like(remaining, -np.inf)
    full_score[mask] = score
    
    return full_score
# EVOLVE-BLOCK-END
