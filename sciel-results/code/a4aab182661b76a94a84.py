"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    The heuristic is formulated as a static, instance-specific 'opportunity cost' matrix.
    Each entry H[i, j] represents the attractiveness of moving from node i to node j.
    
    Following the directive: H[i, j] = prize[j] * (1 - fraction_of_budget_consumed)
    where fraction_of_budget_consumed = (d[i, j] + d[j, 0]) / max_len.
    
    This explicitly penalizes nodes that are too far relative to their value,
    rewarding high-value nodes that leave sufficient slack for the return trip.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # Compute the return cost for each j: d[j, 0]
    return_cost = distance_matrix[:, 0]  # shape (n,)
    
    # For each (i, j), total cost = d[i, j] + d[j, 0]
    total_cost = distance_matrix + return_cost[None, :]  # shape (n, n)
    
    # Fraction of budget consumed
    fraction = total_cost / max_len
    
    # The "remaining fraction" that we multiply by prize
    remaining_fraction = 1.0 - fraction
    
    # Clip to ensure non-negative (infeasible edges get 0, which becomes 1e-9 later)
    remaining_fraction = np.maximum(remaining_fraction, 0.0)
    
    # Heuristic: prize[j] * remaining_fraction
    heuristic = prizes[None, :] * remaining_fraction
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
