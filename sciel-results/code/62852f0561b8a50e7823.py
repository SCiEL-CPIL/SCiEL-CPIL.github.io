import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    # prizes: shape (n,), distance_matrix: shape (n, n)
    n = len(prizes)
    d = distance_matrix
    return_cost = d[:, 0]
    total_cost = d + return_cost[None, :]
    
    # Additive heuristic: Weighted Prize - Weighted Cost
    # This differs from the ratio approach by allowing high prize nodes
    # with moderate cost to be attractive, rather than strictly maximizing
    # prize/cost efficiency.
    prize_weight = 2.0
    cost_weight = 0.5
    
    heuristic = prize_weight * prizes[None, :] - cost_weight * total_cost
    
    # Ensure non-finite and non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    return heuristic
