"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix for ant colony optimization of orienteering.
    
    Uses a blend of:
    - Prize-weighted inverse distance (direct attraction)
    - Prize-weighted eigenvector centrality (structural importance)
    - Distance-decay proximity to origin (feasibility-aware)
    
    The heuristic is an n x n matrix where entry [i, j] represents the
    attractiveness of moving from node i to node j.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Avoid division by zero
    safe_dist = distance_matrix + 1e-9
    
    # Feature 1: Prize-weighted inverse distance
    # High prize at destination, close distance = high attractiveness
    inv_dist = 1.0 / safe_dist
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(inv_dist, 0.0)
    # Attractiveness of moving to j is proportional to prize[j] / dist[i,j]
    feat_prize_dist = inv_dist * prizes[None, :]
    
    # Feature 2: Prize-weighted eigenvector centrality
    # Build a weighted adjacency matrix based on inverse distance
    # Then compute eigenvector centrality weighted by prizes
    adj = inv_dist.copy()
    np.fill_diagonal(adj, 0.0)
    
    # Power iteration for eigenvector centrality
    # Normalize the adjacency matrix
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums
    
    # Start with prize-weighted vector
    eigvec = prizes.copy()
    eigvec[eigvec == 0] = 1e-9
    eigvec = eigvec / (eigvec.sum() + 1e-9)
    
    # Power iteration
    for _ in range(50):
        eigvec_new = adj_norm.T @ eigvec
        eigvec_new = eigvec_new + prizes * 0.1  # Blend with prizes
        eigvec_new[eigvec_new < 0] = 0
        norm = eigvec_new.sum()
        if norm > 0:
            eigvec_new = eigvec_new / norm
        eigvec = eigvec_new
    
    # Create edge feature from eigenvector centrality
    # Attractiveness of j is proportional to eigvec[j]
    feat_eigvec = np.outer(np.ones(n), eigvec)
    
    # Feature 3: Proximity to origin (node 0)
    # Nodes closer to origin are more attractive as they leave more budget
    dist_to_origin = distance_matrix[0, :]
    safe_dist_origin = dist_to_origin + 1e-9
    proximity = 1.0 / safe_dist_origin
    # Normalize
    proximity = proximity / (proximity.sum() + 1e-9)
    
    # Combine features with weights
    # Weight 1: Prize-distance (strong local signal)
    # Weight 2: Eigenvector centrality (structural signal)
    # Weight 3: Proximity to origin (feasibility signal)
    w1 = 0.5
    w2 = 0.3
    w3 = 0.2
    
    heuristic = w1 * feat_prize_dist + w2 * feat_eigvec + w3 * np.outer(np.ones(n), proximity)
    
    # Ensure non-negative
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to a small value (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
