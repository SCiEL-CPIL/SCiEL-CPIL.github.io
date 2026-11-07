"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-by-item matrix of attractiveness for placing item j right after
    item i in the same bin. Uses a complementarity-based design:
    - Pairs that fit and have low waste get high attractiveness.
    - A sharp power emphasizes high fill ratios.
    - A tight-fit bonus gives near-perfect complements distinct advantage.
    - A global structural term rewards pairs that are near-optimal complements
      in a sorted-order two-pointer pairing.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Compute pair sums
    pair = demands[:, None] + demands[None, :]
    
    # Waste: how much capacity is left if both items are in the same bin
    # Only valid where pair <= capacity
    waste = capacity - pair  # positive when pair fits, negative otherwise
    
    # Base heuristic: 1 where it fits, 0 otherwise (will be sharpened)
    fits = pair <= capacity
    
    # Fill ratio for fitting pairs
    fill_ratio = np.zeros((n, n), dtype=np.float64)
    fill_ratio[fits] = pair[fits] / capacity
    
    # Sharp power to emphasize high fill ratios
    # Use a moderate power that gives strong preference to high fill
    power = 12.0
    base = np.zeros((n, n), dtype=np.float64)
    base[fits] = fill_ratio[fits] ** power
    
    # Tight-fit bonus: pairs with very low waste get extra boost
    # waste < 5 means the pair uses at least capacity-5 of the bin
    tight_mask = fits & (waste < 5)
    tight_bonus = np.zeros((n, n), dtype=np.float64)
    tight_bonus[tight_mask] = 2.0 / (waste[tight_mask] + 1.0)
    
    # Global structural term via two-pointer complementarity
    # Sort demands and find ideal partner for each item
    sorted_idx = np.argsort(demands)
    sorted_demands = demands[sorted_idx]
    
    # Two-pointer sweep to find ideal complements
    # For each item, find the largest item that fits with it
    ideal_partner = np.zeros(n, dtype=np.int64)
    left = np.zeros(n, dtype=np.int64)
    right = np.zeros(n, dtype=np.int64)
    
    # For each sorted position, find the best complement
    # Use two pointers: for item at position i, find largest j such that
    # sorted_demands[i] + sorted_demands[j] <= capacity
    # We want to map back to original indices
    best_complement_val = np.zeros(n, dtype=np.float64)
    
    # For each item, compute its ideal complement value
    # An item of size s ideally pairs with an item of size capacity - s
    ideal_complement = capacity - demands
    
    # Structural score: how close is pair (i,j) to being an ideal complement pair?
    # For item i, its ideal partner has size ~ ideal_complement[i]
    # So pair (i,j) is structurally good if demands[j] is close to ideal_complement[i]
    # AND pair (j,i) is structurally good if demands[i] is close to ideal_complement[j]
    
    # Compute structural closeness
    # dev_ij = |demands[j] - ideal_complement[i]| for fitting pairs
    dev_ij = np.abs(demands[None, :] - ideal_complement[:, None])  # n x n
    dev_ji = np.abs(demands[:, None] - ideal_complement[None, :])  # n x n
    
    # Structural score: high when both deviations are small
    # Use a Gaussian-like decay
    sigma = 10.0  # tolerance in demand units
    struct_ij = np.exp(-dev_ij ** 2 / (2 * sigma ** 2))
    struct_ji = np.exp(-dev_ji ** 2 / (2 * sigma ** 2))
    
    # Only apply structural bonus where the pair fits
    struct_score = np.minimum(struct_ij, struct_ji)  # both directions should be good
    struct_score[~fits] = 0.0
    
    # Combine: base + tight_bonus + structural bonus
    # Scale structural term to be comparable
    struct_weight = 1.5
    heuristic = base + tight_bonus + struct_weight * struct_score
    
    # Ensure non-fitting pairs get very low value
    heuristic[~fits] = 0.0
    
    # Replace non-positive or non-finite values with 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Ensure diagonal is low (item can't pair with itself meaningfully in this context)
    # Actually, the ant places item j after item i, so i != j should be handled by the ant logic
    # But let's keep diagonal at minimum to be safe
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
