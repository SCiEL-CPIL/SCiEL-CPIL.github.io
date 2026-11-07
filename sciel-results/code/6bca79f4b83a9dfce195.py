"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using geometric mean of per-constraint
    prize-to-weight ratios, boosted by relative total prize value.
    
    Args:
        prizes: (5, 100) array of prize values
        weights: (5, 100) array of weight values
    
    Returns:
        Array of shape (100,) with one attractiveness score per item.
    """
    # Ensure correct shapes
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != weights.shape:
        raise ValueError("prizes and weights must have the same shape")
    
    n_constraints, n_items = prizes.shape
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    w_safe = weights + eps
    
    # Per-constraint prize-to-weight ratios
    ratios = prizes / w_safe  # shape: (n_constraints, n_items)
    
    # Geometric mean of ratios across constraints for each item
    # log-sum-exp trick for numerical stability
    log_ratios = np.log(ratios)
    geo_mean = np.exp(np.mean(log_ratios, axis=0))  # shape: (n_items,)
    
    # Total prize per item
    total_prize = np.sum(prizes, axis=0)  # shape: (n_items,)
    
    # Normalize total prize to [0, 1] range for boosting
    max_total = np.max(total_prize)
    if max_total > eps:
        rel_prize = total_prize / max_total
    else:
        rel_prize = np.ones(n_items)
    
    # Combine: geometric mean efficiency boosted by relative prize value
    # The boost makes high-value items more attractive even if slightly less efficient
    heuristic = geo_mean * (1.0 + 2.0 * rel_prize)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
