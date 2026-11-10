"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization
    using a learned linear combination of graph-theoretic centrality features.
    
    Features used:
    1. Prize-weighted eigenvector centrality
    2. Degree centrality (inverse distance sum)
    3. Betweenness-like centrality (proximity to many pairs)
    4. Edge feasibility (distance constraint satisfaction)
    
    The heuristic is a weighted combination designed to guide ants toward
    high-value nodes that are structurally central in the graph.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure no negative distances
    distance_matrix = np.maximum(distance_matrix, 0.0)
    
    eps = 1e-9
    
    # Feature 1: Prize-weighted eigenvector centrality
    # Weighted adjacency: W[i,j] = prizes[j] / (d[i,j] + eps)
    # This emphasizes high prize neighbors
    inv_dist = np.zeros((n, n))
    mask = ~np.eye(n, dtype=bool)
    inv_dist[mask] = 1.0 / (distance_matrix[mask] + eps)
    weighted_adj = inv_dist * prizes[None, :] # Each column j scaled by prize[j]
    
    eigvec = np.ones(n) / n
    for _ in range(50): # Reduced iterations for speed, convergence is fast
        new_eigvec = weighted_adj @ eigvec
        norm = np.sum(new_eigvec)
        if norm > 0:
            new_eigvec = new_eigvec / norm
        else:
            break
        if np.allclose(new_eigvec, eigvec, atol=1e-8):
            break
        eigvec = new_eigvec
        
    # Normalize eigvec to [0, 1]
    eig_min = np.min(eigvec)
    eig_max = np.max(eigvec)
    if eig_max > eig_min:
        eig_norm = (eigvec - eig_min) / (eig_max - eig_min)
    else:
        eig_norm = np.ones(n) / n

    # Feature 2: Degree centrality (inverse distance sum)
    degree_centrality = np.sum(inv_dist, axis=1)
    deg_min = np.min(degree_centrality)
    deg_max = np.max(degree_centrality)
    if deg_max > deg_min:
        deg_norm = (degree_centrality - deg_min) / (deg_max - deg_min)
    else:
        deg_norm = np.ones(n) / n

    # Feature 3: Betweenness-like centrality
    # Approximation: For each node k, count pairs (i, j) such that d[i,k] + d[k,j] is close to d[i,j]
    # This identifies nodes that are "intermediate" or "central" in the metric space.
    betweenness = np.zeros(n)
    # Vectorized calculation of path lengths through k
    # d[i,k] + d[k,j] for all i, j, k
    # We can loop over k (n=50) and vectorize i, j
    for k in range(n):
        # d[i, k] is a column vector (n, 1)
        # d[k, j] is a row vector (1, n)
        path_via_k = distance_matrix[:, k:k+1] + distance_matrix[k:k+1, :]
        direct = distance_matrix
        # Check if path_via_k is slightly longer than direct (within tolerance)
        # Tolerance: 1.02 * direct to account for floating point and slight detours
        # Actually, for "betweenness" in a complete graph, we want nodes that are
        # "on the way". If d[i,k] + d[k,j] < d[i,j] + tol, k is a candidate.
        # In Euclidean space, d[i,k] + d[k,j] >= d[i,j] always.
        # So we check if d[i,k] + d[k,j] - d[i,j] < tol.
        tol = 0.05 * np.max(distance_matrix) # Relative tolerance
        slack = path_via_k - direct
        # Count where slack is small
        # Exclude i=k or j=k
        mask_k = np.ones((n, n), dtype=bool)
        mask_k[k, :] = False
        mask_k[:, k] = False
        count = np.sum((slack < tol) & mask_k)
        betweenness[k] = count
        
    bet_min = np.min(betweenness)
    bet_max = np.max(betweenness)
    if bet_max > bet_min:
        bet_norm = (betweenness - bet_min) / (bet_max - bet_min)
    else:
        bet_norm = np.ones(n) / n

    # Feature 4: Prize (normalized)
    prize_min = np.min(prizes)
    prize_max = np.max(prizes)
    if prize_max > prize_min:
        prize_norm = (prizes - prize_min) / (prize_max - prize_min)
    else:
        prize_norm = np.ones(n) / n

    # Build edge-wise heuristic
    # Heuristic[i, j] = w1 * Prize[j] + w2 * Eig[j] + w3 * Deg[j] + w4 * Bet[j] + w5 * Feas[i, j]
    
    # Feasibility: How much room is left after going i->j->0?
    # Cost = d[i,j] + d[j,0]
    # Feas = 1 - Cost / max_len (clipped to 0)
    cost_matrix = distance_matrix + distance_matrix[:, 0:1] # d[i,j] + d[j,0]
    feas_matrix = 1.0 - cost_matrix / max_len
    feas_matrix = np.clip(feas_matrix, 0.0, 1.0)
    # Only valid if i != j
    feas_matrix[~mask] = 0.0

    # Weights: Emphasize Prize and Structural Centrality (Betweenness/Eigenvector)
    w_prize = 0.35
    w_eig = 0.25
    w_deg = 0.15
    w_bet = 0.15
    w_feas = 0.10

    # Construct matrix
    # Node features are broadcast to columns
    h_prize = w_prize * prize_norm[None, :]
    h_eig = w_eig * eig_norm[None, :]
    h_deg = w_deg * deg_norm[None, :]
    h_bet = w_bet * bet_norm[None, :]
    
    heuristic = h_prize + h_eig + h_deg + h_bet + w_feas * feas_matrix
    
    # Zero out diagonal
    heuristic[~mask] = 0.0
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
