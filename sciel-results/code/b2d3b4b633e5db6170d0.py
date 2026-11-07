"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Uses geometric mean of per-constraint prize-to-weight ratios as a
    smooth surrogate for the bottleneck constraint, favoring items that
    are efficient across ALL constraints rather than just in aggregate.
    """
    eps = 1e-9
    # Per-constraint prize/weight ratio for each item
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean across constraints: (prod(ratios_c))^(1/5)
    # Using log-space for numerical stability
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    
    # Exponentiate back to get the geometric mean
    geo_mean = np.exp(log_geo_mean)
    
    # Ensure all values are positive (they should be, but guard against underflow)
    geo_mean = np.maximum(geo_mean, 1e-9)
    
    return geo_mean
# EVOLVE-BLOCK-END
