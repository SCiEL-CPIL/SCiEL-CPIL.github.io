"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic edge-attractiveness matrix for ACO-CVRP.
    
    Combines:
    1. Inverse distance (base attractiveness)
    2. Pair-wise demand compatibility penalty (high-demand pairs are less attractive)
    3. Depot proximity bias (nodes close to depot get boosted depot edges)
    4. Local spatial coherence (edges aligned with local neighborhood flow get boosted)
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = d_mat.shape[0]
    cap = float(capacity)
    
    epsilon = 1e-9
    
    # 1. Base: inverse distance
    base = 1.0 / (d_mat + epsilon)
    np.fill_diagonal(base, 0.0)
    
    # 2. Pair-wise demand compatibility penalty
    # Edges between two high-demand nodes are less attractive because
    # it's harder to fit both on the same route with other customers.
    # Use a smooth penalty based on the sum of demands relative to capacity.
    demand_sum = dem[:, np.newaxis] + dem[np.newaxis, :]
    # Normalize: ratio = demand_sum / cap, range roughly [2/cap, 18/cap]
    # Penalize when ratio > 1.0 (combined demand exceeds capacity)
    # Smooth penalty: 1 / (1 + gamma * max(0, ratio - 0.8)^2)
    ratio = demand_sum / cap
    excess = np.maximum(0.0, ratio - 0.8)
    gamma = 2.0
    demand_penalty = 1.0 / (1.0 + gamma * excess ** 2)
    
    heuristic = base * demand_penalty
    
    # 3. Depot proximity bias
    # Nodes closer to depot (index 0) get a boost on edges involving depot.
    # This encourages ants to start/end routes with nearby customers,
    # reducing the "tail" cost of returning to depot.
    dist_to_depot = d_mat[0, :]
    min_d = dist_to_depot.min()
    max_d = dist_to_depot.max()
    if max_d > min_d:
        norm_dist = (dist_to_depot - min_d) / (max_d - min_d)
    else:
        norm_dist = np.zeros(n)
    
    # Close nodes get higher bonus: 1 + alpha * (1 - norm_dist)
    alpha_depot = 0.4
    depot_factor = 1.0 + alpha_depot * (1.0 - norm_dist)
    
    # Apply to depot row and column only
    depot_bonus = np.ones((n, n))
    depot_bonus[0, :] = depot_factor
    depot_bonus[:, 0] = depot_factor
    np.fill_diagonal(depot_bonus, 1.0)
    
    heuristic = heuristic * depot_bonus
    
    # 4. Local spatial coherence (simplified)
    # For each node i, find k nearest neighbors and compute the mean direction.
    # Boost edges from i that align with this mean direction.
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
    # alignment[i, j] = dot(mean_dirs_norm[i], unit_vector(i -> j))
    edge_vecs = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (n, n, 2)
    edge_norms = np.linalg.norm(edge_vecs, axis=2)
    edge_norms_safe = np.where(edge_norms > epsilon, edge_norms, epsilon)
    edge_dirs = edge_vecs / edge_norms_safe[:, :, np.newaxis]
    
    # Dot product: for each i, dot(mean_dirs_norm[i], edge_dirs[i, :])
    alignment = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        alignment[i, :] = np.sum(mean_dirs_norm[i] * edge_dirs[i, :], axis=1)
    
    # Mild boost for positive alignment
    beta_spatial = 0.3
    spatial_boost = 1.0 + beta_spatial * np.maximum(0.0, alignment)
    
    heuristic = heuristic * spatial_boost
    
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
