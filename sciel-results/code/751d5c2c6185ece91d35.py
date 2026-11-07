import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ant colony orienteering.
    
    Uses prize efficiency with return-cost awareness:
    eta[i,j] = (prize[j] / (d[i,j] + d[j,0] + eps))^gamma
    
    This accounts for the fact that visiting node j incurs a cost of
    d[i,j] now plus d[j,0] as a future return liability, which directly
    impacts the max_len constraint.
    """
    n = len(prizes)
    eps = 1e-9
    gamma = 2.0
    
    # d[j,0] is the return cost from node j to start
    d_to_start = distance_matrix[:, 0].copy()  # shape (n,)
    
    # For each pair (i, j), the effective cost is d[i,j] + d[j,0]
    # eff_cost[i, j] = distance_matrix[i, j] + d_to_start[j]
    eff_cost = distance_matrix + d_to_start[None, :]  # shape (n, n)
    
    # Prize efficiency: prize[j] / eff_cost[i,j]
    # Avoid division by zero
    eff_cost_safe = eff_cost + eps
    
    # eta[i,j] = (prize[j] / eff_cost_safe[i,j])^gamma
    prize_row = prizes[None, :]  # shape (1, n)
    eta = (prize_row / eff_cost_safe) ** gamma
    
    # Set eta[i,0] = 1e-9 for all i (no point visiting node 0)
    eta[:, 0] = 1e-9
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
