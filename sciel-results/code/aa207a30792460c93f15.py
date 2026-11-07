import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ant colony orienteering.
    
    Uses a combined heuristic that balances:
    1. Prize efficiency: prize[j] / (d[i,j] + d[j,0])
    2. Distance penalty: penalize long detours more strongly
    3. Position-awareness: prefer nodes that keep the path short
    
    eta[i,j] = (prize[j] / (d[i,j] + d[j,0] + eps))^gamma * (1 / (1 + d[i,j]/max_len))^beta
    """
    n = len(prizes)
    eps = 1e-9
    gamma = 2.5
    beta = 1.5
    
    # d[j,0] is the return cost from node j to start
    d_to_start = distance_matrix[:, 0].copy()  # shape (n,)
    
    # For each pair (i, j), the effective cost is d[i,j] + d[j,0]
    eff_cost = distance_matrix + d_to_start[None, :]  # shape (n, n)
    eff_cost_safe = eff_cost + eps
    
    # Prize efficiency component
    prize_row = prizes[None, :]  # shape (1, n)
    prize_eff = (prize_row / eff_cost_safe) ** gamma
    
    # Distance penalty: penalize edges that are long relative to max_len
    # This helps ants avoid moves that consume too much of the remaining budget
    dist_penalty = (1.0 / (1.0 + distance_matrix / max_len)) ** beta
    
    # Combine: multiply prize efficiency by distance penalty
    eta = prize_eff * dist_penalty
    
    # Set eta[i,0] = 1e-9 for all i (no point visiting node 0)
    eta[:, 0] = 1e-9
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
