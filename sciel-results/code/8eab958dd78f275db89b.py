"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic edge-attractiveness matrix for ACO-CVRP.
    
    Combines:
    1. Steep inverse distance (1/d^2) for tight clustering
    2. Angular continuity bonus (non-crossing, sector-based routes)
    3. Pair-wise demand compatibility penalty
    4. Depot proximity bias
    5. Local spatial coherence
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = d_mat.shape[0]
    cap = float(capacity)
    
    epsilon = 1e-9
    
    # 1. Base: steep inverse distance (1/d^2) for tighter clustering
    base = 1.0 / (d_mat ** 2 + epsilon)
    np.fill_diagonal(base, 0.0)
    
    # 2. Pair-wise demand compatibility penalty
    demand_sum = dem[:, np.newaxis] + dem[np.newaxis, :]
    ratio = demand_sum / cap
    excess = np.maximum(0.0, ratio - 0.8)
    gamma = 2.0
    demand_penalty = 1.0 / (1.0 + gamma * excess ** 2)
    
    heuristic = base * demand_penalty
    
    # 3. Depot proximity bias
    dist_to_depot = d_mat[0, :]
    min_d = dist_to_depot.min()
    max_d = dist_to_depot.max()
    if max_d > min_d:
        norm_dist = (dist_to_depot - min_d) / (max_d - min_d)
    else:
        norm_dist = np.zeros(n)
    
    alpha_depot = 0.4
    depot_factor = 1.0 + alpha_depot * (1.0 - norm_dist)
    
    depot_bonus = np.ones((n, n))
    depot_bonus[0, :] = depot_factor
    depot_bonus[:, 0] = depot_factor
    np.fill_diagonal(depot_bonus, 1.0)
    
    heuristic = heuristic * depot_bonus
    
    # 4. Local spatial coherence with angular continuity
    k = min(5, n - 1)
    
    # Compute k-NN indices for each node
    nn_indices = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        dists = d_mat[i].copy()
        dists[i] = np.inf
        nn_indices[i] = np.argpartition(dists, k)[:k]
    
    # Compute mean direction to k-NN for each node
    mean_dirs = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        nn = nn_indices[i]
        vecs = coords[nn] - coords[i]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < epsilon] = epsilon
        vecs_norm = vecs / norms
        mean_dirs[i] = np.mean(vecs_norm, axis=0)
    
    # Normalize mean directions
    md_norms = np.linalg.norm(mean_dirs, axis=1, keepdims=True)
    md_norms[md_norms < epsilon] = epsilon
    mean_dirs_norm = mean_dirs / md_norms
    
    # Compute alignment for each edge (i, j)
    edge_vecs = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (n, n, 2)
    edge_norms = np.linalg.norm(edge_vecs, axis=2)
    edge_norms_safe = np.where(edge_norms > epsilon, edge_norms, epsilon)
    edge_dirs = edge_vecs / edge_norms_safe[:, :, np.newaxis]
    
    # Dot product: for each i, dot(mean_dirs_norm[i], edge_dirs[i, :])
    alignment = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        alignment[i, :] = np.sum(mean_dirs_norm[i] * edge_dirs[i, :], axis=1)
    
    # Angular continuity: boost edges that maintain direction consistency
    # For a route i -> j -> k, the angle at j should be small (non-crossing)
    # We approximate this by boosting edges (i,j) where j's mean direction
    # aligns well with the direction from i to j
    
    # Stronger angular continuity bonus
    beta_angular = 0.8
    angular_boost = 1.0 + beta_angular * np.maximum(0.0, alignment)
    
    heuristic = heuristic * angular_boost
    
    # 5. Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 6. Normalize so median is ~1.0 for numerical stability
    positive_vals = heuristic[heuristic > 0]
    if len(positive_vals) > 0:
        med = np.median(positive_vals)
        if med > 0:
            heuristic = heuristic / med
    
    # Final safety
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
