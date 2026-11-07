"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: Array of shape (C, N) or (N,), where C is constraints and N is items.
        weights: Array of shape (C, N).
        
    Returns:
        Array of shape (N,) with heuristic scores.
    """
    # Ensure prizes is treated as having the same number of rows as weights if needed for broadcasting
    # If prizes is 1D (N,), it broadcasts against weights (C, N).
    # If prizes is 2D (C, N), it matches weights (C, N).
    
    # Calculate value per item. If prizes is 2D, sum across constraints to get a total scalar value per item.
    # If prizes is 1D, it is already the scalar value.
    if prizes.ndim == 1:
        values = prizes
    else:
        values = prizes.sum(axis=0)
        
    # Add epsilon to weights to avoid division by zero
    eps = 1e-9
    safe_weights = weights + eps
    
    # Calculate ratio of value to weight for each constraint
    # ratios has shape (C, N)
    ratios = values / safe_weights
    
    # The heuristic is the minimum ratio across all constraints for each item.
    # This ensures an item is only highly attractive if it is efficient in ALL constraints,
    # specifically its worst-case constraint. This "bottleneck" approach is robust for MCKP.
    scores = ratios.min(axis=0)
    
    # Ensure scores are positive for the ACO (though min of positive/positive is positive)
    # Replace any non-finite or non-positive values with a small number
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
