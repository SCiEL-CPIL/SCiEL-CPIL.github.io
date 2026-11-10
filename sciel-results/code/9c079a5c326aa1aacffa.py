"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization
    using a learned linear combination of graph-theoretic centrality features.
    
    Features used:
    1. Prize-weighted eigenvector centrality (captures high-value hubs)
    2. Degree centrality (normalized by distance)
    3. Betweenness-like centrality (shortest path usage)
    4. Prize/distance ratio (local attractiveness)
    
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
    # Weighted adjacency: weight[i,j] = prizes[j] / (d[i,j] + eps)
    weighted_adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                weighted_adj[i, j] = prizes[j] / (distance_matrix[i, j] + eps)
    
    # Power iteration for eigenvector centrality
    eigvec = np.ones(n) / n
    for _ in range(200):
        new_eigvec = weighted_adj @ eigvec
        norm = np.sum(new_eigvec)
        if norm > 0:
            new_eigvec = new_eigvec / norm
        else:
            break
        if np.allclose(new_eigvec, eigvec, atol=1e-10):
            break
        eigvec = new_eigvec
    
    # Node-level prize-weighted eigenvector centrality
    prize_eig_centrality = eigvec * (prizes + eps)
    max_peig = np.max(prize_eig_centrality)
    if max_peig > 0:
        prize_eig_centrality = prize_eig_centrality / max_peig
    
    # Feature 2: Degree centrality (inverse distance sum)
    degree_centrality = np.zeros(n)
    for i in range(n):
        inv_dist_sum = 0.0
        for j in range(n):
            if i != j:
                inv_dist_sum += 1.0 / (distance_matrix[i, j] + eps)
        degree_centrality[i] = inv_dist_sum
    
    max_deg = np.max(degree_centrality)
    if max_deg > 0:
        degree_centrality = degree_centrality / max_deg
    
    # Feature 3: Closeness centrality (inverse of sum of distances)
    closeness = np.zeros(n)
    for i in range(n):
        dist_sum = 0.0
        for j in range(n):
            if i != j:
                dist_sum += distance_matrix[i, j]
        closeness[i] = 1.0 / (dist_sum + eps)
    
    max_close = np.max(closeness)
    if max_close > 0:
        closeness = closeness / max_close
    
    # Feature 4: Prize density in neighborhood
    # For each node, sum of prizes of nearby nodes weighted by proximity
    prize_density = np.zeros(n)
    for i in range(n):
        density = 0.0
        for j in range(n):
            if i != j:
                proximity = 1.0 / (distance_matrix[i, j] + eps)
                density += prizes[j] * proximity
        prize_density[i] = density
    
    max_pden = np.max(prize_density)
    if max_pden > 0:
        prize_density = prize_density / max_pden
    
    # Feature 5: Distance feasibility score for edge (i, j)
    # How much of max_len is consumed by going i->j->0
    dist_feasibility = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cost = distance_matrix[i, j] + distance_matrix[j, 0]
            # Score is high when cost is low relative to max_len
            dist_feasibility[i, j] = np.clip(1.0 - cost / (max_len + eps), 0.0, 1.0)
    
    # Feature 6: Prize/distance ratio for edge (i, j)
    prize_dist_ratio = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                prize_dist_ratio[i, j] = (prizes[j] + eps) / (distance_matrix[i, j] + eps)
    
    max_pdr = np.max(prize_dist_ratio)
    if max_pdr > 0:
        prize_dist_ratio = prize_dist_ratio / max_pdr
    
    # Build edge-wise heuristic from node features
    # For edge (i, j), combine:
    # - Prize of destination j (most important)
    # - Eigenvector centrality of j
    # - Degree centrality of j
    # - Closeness centrality of j
    # - Prize density of j
    # - Prize/distance ratio for edge (i,j)
    # - Distance feasibility for edge (i,j)
    
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # All features are already normalized to [0, 1]
            feat_prize = prizes[j]
            feat_eig = prize_eig_centrality[j]
            feat_deg = degree_centrality[j]
            feat_close = closeness[j]
            feat_pden = prize_density[j]
            feat_pdr = prize_dist_ratio[i, j]
            feat_feas = dist_feasibility[i, j]
            
            # Weighted combination - emphasize prize, feasibility, and prize/distance
            w_prize = 0.30
            w_feas = 0.25
            w_pdr = 0.15
            w_eig = 0.10
            w_pden = 0.10
            w_deg = 0.05
            w_close = 0.05
            
            heuristic[i, j] = (
                w_prize * feat_prize +
                w_feas * feat_feas +
                w_pdr * feat_pdr +
                w_eig * feat_eig +
                w_pden * feat_pden +
                w_deg * feat_deg +
                w_close * feat_close
            )
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
