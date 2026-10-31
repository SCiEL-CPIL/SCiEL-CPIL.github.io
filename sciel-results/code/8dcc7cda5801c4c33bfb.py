"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit.
    We choose the bin with the maximum remaining space after placing the item.
    This heuristic preserves capacity in other bins for potentially larger
    future items, which can be beneficial for distributions with a long tail
    (such as Weibull), as it prevents fragmentation where small items are
    stuck in bins that are too full for them but too empty for large ones.
    """
    eps = 1e-9
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Worst Fit: Maximize the remaining space
        scores[feasible] = remaining[feasible]
        
    return scores
# EVOLVE-BLOCK-END
