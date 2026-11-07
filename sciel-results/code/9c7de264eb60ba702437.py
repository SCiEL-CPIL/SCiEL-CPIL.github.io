"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    Strategy: Detour-aware efficiency with cluster-density bonus.
    For each edge (i, j), the heuristic encodes:
      1. Prize-per-detour-cost: prize_j / (d[i,j] + d[j,0] + eps), amplified by power.
      2. Cluster value: expected prize reachable from j within remaining budget.
      3. Origin proximity: bonus for nodes close to origin (easier return).
    """
    start_time = time.time()
    DEADLINE = 280.0  # safety margin before 330s kill
    
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    
    eps = 1e-9
    d_to_origin = dist[:, 0].copy()  # d[j, 0] for all j
    
    # --- Feature 1: Detour-aware efficiency ---
    # Full cost of visiting j from i and returning: d[i,j] + d[j,0]
    detour_cost = dist + d_to_origin[None, :]  # shape (n, n)
    # Prize per unit detour cost, amplified
    gamma = 2.0
    efficiency = (prizes[None, :] + eps) / (detour_cost + eps)
    efficiency = efficiency ** gamma
    
    # --- Feature 2: Cluster value (prize-weighted density within budget) ---
    # For each node j, compute the total prize of nodes k that are "reachable"
    # after visiting j, i.e., d[j,k] + d[k,0] <= max_len - d[j,0] (roughly)
    # Simplified: prize-weighted count of neighbors within a radius proportional to remaining budget
    # Use a kernel: weight = prize_k * exp(-d[j,k]^2 / (2*sigma^2)) where sigma ~ max_len/3
    sigma = max_len / 3.0
    # For each j, sum over k of prize_k * exp(-d[j,k]^2 / (2*sigma^2))
    # This is a Gaussian-weighted prize density
    dist_sq = dist ** 2
    kernel = np.exp(-dist_sq / (2.0 * sigma * sigma))
    cluster_value = kernel @ (prizes + eps)  # shape (n,)
    
    # Normalize cluster value to [0, 1] range for stable combination
    cv_min = cluster_value.min()
    cv_max = cluster_value.max()
    if cv_max > cv_min:
        cluster_norm = (cluster_value - cv_min) / (cv_max - cv_min)
    else:
        cluster_norm = np.ones(n)
    
    # Cluster bonus as multiplicative factor: 1 + alpha * cluster_norm
    alpha = 0.5
    cluster_bonus = 1.0 + alpha * cluster_norm[None, :]  # shape (n, n)
    
    # --- Feature 3: Origin proximity bonus ---
    # Nodes closer to origin are easier to return to, especially when path is long
    # Normalize d_to_origin to [0, 1]
    dto_min = d_to_origin.min()
    dto_max = d_to_origin.max()
    if dto_max > dto_min:
        dto_norm = (d_to_origin - dto_min) / (dto_max - dto_min)
    else:
        dto_norm = np.zeros(n)
    
    # Bonus: closer to origin gets higher multiplier
    beta = 0.3
    origin_bonus = 1.0 + beta * (1.0 - dto_norm)[None, :]  # shape (n, n)
    
    # --- Combine features ---
    # heuristic[i,j] = efficiency[i,j] * cluster_bonus[i,j] * origin_bonus[i,j]
    heuristic = efficiency * cluster_bonus * origin_bonus
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return heuristic
    
    # Verify shape
    if heuristic.shape != (n, n):
        heuristic = np.ones((n, n)) * 1e-9
    
    return heuristic
