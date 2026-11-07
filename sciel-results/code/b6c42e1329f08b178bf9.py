"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Uses a soft-min of per-constraint efficiencies (prize/weight) as the heuristic.
    The soft-min provides a smooth surrogate for the bottleneck efficiency,
    preferring items that are efficient across all constraints while being
    less cliff-like than a hard min.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Per-constraint efficiency: prize / weight for each constraint and item
    eff = prizes / (weights + 1e-9)
    
    # Soft-min of efficiencies across constraints using log-sum-exp
    # soft_min(x, T) = min(x) - T * log(sum(exp(-(x - min(x)) / T)))
    T = 0.5  # Temperature: smaller = closer to hard min, larger = closer to mean
    
    min_eff = eff.min(axis=0)
    # For numerical stability, shift by min before exponentiating
    shifted = (eff - min_eff[:, np.newaxis]) / T
    # exp(-shifted) where shifted >= 0, so values are in (0, 1]
    exp_neg_shifted = np.exp(-shifted)
    sum_exp = exp_neg_shifted.sum(axis=0)
    
    soft_min_eff = min_eff - T * np.log(sum_exp)
    
    # The soft-min is slightly less than the hard min (more negative penalty),
    # but preserves the relative ordering. Use it as the heuristic.
    h = soft_min_eff
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
