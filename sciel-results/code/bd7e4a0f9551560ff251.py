"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-by-item matrix of attractiveness for placing item j right after
    item i in the same bin. Uses a complementarity-based design with smooth
    fill-ratio emphasis and tight-fit bonus.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Compute pair sums
    pair = demands[:, None] + demands[None, :]
    
    # Waste: how much capacity is left if both items are in the same bin
    waste = capacity - pair
    
    # Base heuristic: only valid where pair fits
    fits = pair <= capacity
    
    # Fill ratio for fitting pairs
    fill_ratio = np.zeros((n, n), dtype=np.float64)
    fill_ratio[fits] = pair[fits] / capacity
    
    # Smooth power to emphasize high fill ratios
    # fill_ratio^2 + 0.1 gives a smoother, more discriminating heuristic
    power = 2.0
    base = np.zeros((n, n), dtype=np.float64)
    base[fits] = fill_ratio[fits] ** power + 0.1
    
    # Tight-fit bonus: pairs with very low waste get extra boost
    # Use a wider threshold and stronger weighting
    tight_mask = fits & (waste < 10)
    tight_bonus = np.zeros((n, n), dtype=np.float64)
    tight_bonus[tight_mask] = 3.0 / (waste[tight_mask] + 1.0)
    
    # Global structural term via complementarity
    ideal_complement = capacity - demands
    
    # Compute structural closeness
    dev_ij = np.abs(demands[None, :] - ideal_complement[:, None])
    dev_ji = np.abs(demands[:, None] - ideal_complement[None, :])
    
    # Structural score with tighter sigma for more selective pairing
    sigma = 8.0
    struct_ij = np.exp(-dev_ij ** 2 / (2 * sigma ** 2))
    struct_ji = np.exp(-dev_ji ** 2 / (2 * sigma ** 2))
    
    # Only apply structural bonus where the pair fits
    struct_score = np.minimum(struct_ij, struct_ji)
    struct_score[~fits] = 0.0
    
    # Combine: base + tight_bonus + structural bonus
    struct_weight = 1.0
    heuristic = base + tight_bonus + struct_weight * struct_score
    
    # Ensure non-fitting pairs get very low value
    heuristic[~fits] = 0.0
    
    # Replace non-positive or non-finite values with 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Ensure diagonal is low
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
