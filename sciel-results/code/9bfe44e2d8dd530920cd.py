"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization
    using a linear combination of features tailored for Orienteering with Prizes.
    
    Features:
    1. Prize of destination node j.
    2. Feasibility: Remaining budget if we go i->j->0.
    3. Proximity: Inverse distance i->j.
    4. Return Proximity: Inverse distance j->0.
    5. Prize Density: Prize of j divided by total cost to include j (i->j->0).
    
    Weights are tuned to balance high prize collection with route feasibility.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure no negative distances
    distance_matrix = np.maximum(distance_matrix, 0.0)
    eps = 1e-9
    
    # Precompute node-level features
    # d_j0: distance from node j to depot (node 0)
    d_j0 = distance_matrix[:, 0]
    
    # Initialize feature matrices
    # F1: Prize (node j)
    F1 = np.tile(prizes, (n, 1))
    
    # F2: Feasibility (edge i->j)
    # Cost = d[i,j] + d[j,0]
    # Feasibility is high if cost is low relative to max_len
    # We use 1 - cost/max_len, clipped to [0, 1]
    costs = distance_matrix + d_j0[np.newaxis, :]
    F2 = 1.0 - costs / max_len
    F2 = np.clip(F2, 0.0, 1.0)
    
    # F3: Proximity (edge i->j)
    # Inverse distance, normalized
    F3 = 1.0 / (distance_matrix + eps)
    # Normalize F3 to [0, 1]
    F3_min = np.min(F3)
    F3_max = np.max(F3)
    if F3_max > F3_min:
        F3 = (F3 - F3_min) / (F3_max - F3_min)
    else:
        F3 = np.ones_like(F3)
        
    # F4: Return Proximity (node j)
    # Inverse distance to depot, normalized
    F4_node = 1.0 / (d_j0 + eps)
    F4_node_min = np.min(F4_node)
    F4_node_max = np.max(F4_node)
    if F4_node_max > F4_node_min:
        F4_node = (F4_node - F4_node_min) / (F4_node_max - F4_node_min)
    else:
        F4_node = np.ones_like(F4_node)
    F4 = np.tile(F4_node, (n, 1))
    
    # F5: Prize Density (edge i->j)
    # Prize of j / Cost(i->j->0)
    F5 = prizes[np.newaxis, :] / (costs + eps)
    F5_min = np.min(F5)
    F5_max = np.max(F5)
    if F5_max > F5_min:
        F5 = (F5 - F5_min) / (F5_max - F5_min)
    else:
        F5 = np.ones_like(F5)
        
    # Linear Combination
    # Weights:
    # w_prize: Importance of the prize itself.
    # w_feas: Importance of staying within length limit.
    # w_prox: Importance of short steps.
    # w_ret: Importance of being close to depot.
    # w_dens: Importance of value per unit length.
    
    # These weights are chosen to prioritize feasible, high-value moves.
    w_prize = 0.30
    w_feas = 0.25
    w_prox = 0.15
    w_ret = 0.15
    w_dens = 0.15
    
    heuristic = (
        w_prize * F1 +
        w_feas * F2 +
        w_prox * F3 +
        w_ret * F4 +
        w_dens * F5
    )
    
    # Ensure diagonal is low (ants don't stay in place)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
