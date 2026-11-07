"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic edge-attractiveness matrix for ACO-CVRP.
    
    Strategy: Simulate the expected pheromone reinforcement trajectory
    to create a static heuristic that anticipates colony convergence.
    Uses a cooling schedule on the distance exponent to balance
    exploration (early) and exploitation (late).
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = d_mat.shape[0]
    cap = float(capacity)
    
    epsilon = 1e-9
    deadline = time.time() + 25.0  # Safety margin before 300s limit
    
    # 1. Compute base distance heuristic with adaptive exponent
    #    Start with a moderate exponent and cool down
    #    This simulates the transition from exploration to exploitation
    
    # Distance matrix with zero diagonal handled
    d_safe = d_mat.copy()
    np.fill_diagonal(d_safe, 1.0)  # Avoid division by zero, will be masked later
    
    # Simulate a "virtual" ACO run to estimate pheromone accumulation
    # We use a simplified model: edges that are part of many short routes
    # get higher heuristic values
    
    # Step 1: Compute a "route quality" score for each node pair
    # An edge (i,j) is good if:
    # a) It's short (low distance)
    # b) It connects nodes with compatible demands
    # c) It maintains spatial coherence (non-crossing)
    
    # Demand compatibility: penalize pairs that would exceed capacity
    demand_sum = dem[:, np.newaxis] + dem[np.newaxis, :]
    # Normalize by capacity
    demand_ratio = demand_sum / cap
    # Penalize if sum > capacity (shouldn't happen in valid routes, but for heuristic)
    demand_penalty = np.where(demand_ratio > 1.0, 0.5, 1.0)
    # Soft penalty for high demand sums (encourage balancing)
    demand_balance = 1.0 / (1.0 + 0.5 * np.maximum(0, demand_ratio - 0.5))
    
    # Step 2: Spatial coherence via angular continuity
    # For each node, compute the mean direction to its k nearest neighbors
    k = min(6, n - 1)
    
    nn_indices = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        dists = d_mat[i].copy()
        dists[i] = np.inf
        nn_indices[i] = np.argpartition(dists, k)[:k]
    
    # Compute mean direction vectors
    mean_dirs = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        nn = nn_indices[i]
        vecs = coords[nn] - coords[i]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < epsilon] = epsilon
        vecs_norm = vecs / norms
        mean_dirs[i] = np.mean(vecs_norm, axis=0)
    
    md_norms = np.linalg.norm(mean_dirs, axis=1, keepdims=True)
    md_norms[md_norms < epsilon] = epsilon
    mean_dirs_norm = mean_dirs / md_norms
    
    # Edge direction vectors
    edge_vecs = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
    edge_norms = np.linalg.norm(edge_vecs, axis=2)
    edge_norms_safe = np.where(edge_norms > epsilon, edge_norms, epsilon)
    edge_dirs = edge_vecs / edge_norms_safe[:, :, np.newaxis]
    
    # Alignment score: how well edge (i,j) aligns with node i's mean direction
    alignment = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        alignment[i, :] = np.sum(mean_dirs_norm[i] * edge_dirs[i, :], axis=1)
    
    # Angular continuity boost
    angular_boost = 1.0 + 0.6 * np.maximum(0.0, alignment)
    
    # Step 3: Depot proximity bias
    dist_to_depot = d_mat[0, :]
    min_d = dist_to_depot.min()
    max_d = dist_to_depot.max()
    if max_d > min_d:
        norm_dist = (dist_to_depot - min_d) / (max_d - min_d)
    else:
        norm_dist = np.zeros(n)
    
    depot_factor = 1.0 + 0.3 * (1.0 - norm_dist)
    depot_bonus = np.ones((n, n))
    depot_bonus[0, :] = depot_factor
    depot_bonus[:, 0] = depot_factor
    np.fill_diagonal(depot_bonus, 1.0)
    
    # Step 4: Combine with adaptive distance exponent
    # Simulate cooling: start with exponent 1.5, cool to 1.0
    # This creates a heuristic that's less greedy than 1/d^2 but more than 1/d
    
    # Use a weighted combination of different exponents to simulate trajectory
    # Early phase: higher exponent (more local)
    # Late phase: lower exponent (more global)
    
    # Compute multiple distance-based heuristics
    exp_early = 1.8
    exp_late = 1.2
    
    base_early = 1.0 / (d_safe ** exp_early + epsilon)
    base_late = 1.0 / (d_safe ** exp_late + epsilon)
    
    # Weight: 0.4 early, 0.6 late (bias toward exploitation)
    base = 0.4 * base_early + 0.6 * base_late
    np.fill_diagonal(base, 0.0)
    
    # Combine all components
    heuristic = base * demand_balance * angular_boost * depot_bonus
    
    # Step 5: Apply pheromone-like reinforcement
    # Edges that appear in many "good" partial routes get boosted
    # Approximate by: for each node, boost edges to its closest feasible neighbors
    
    # Compute a "local optimality" score: how many of j's neighbors are also i's neighbors
    # This encourages edges that are part of dense clusters
    neighbor_overlap = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Count common neighbors
            common = np.intersect1d(nn_indices[i], nn_indices[j])
            neighbor_overlap[i, j] = len(common) / k
    
    # Boost edges with high neighbor overlap (dense cluster edges)
    overlap_boost = 1.0 + 0.3 * neighbor_overlap
    heuristic = heuristic * overlap_boost
    
    # Step 6: Normalize and ensure positivity
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so median is ~1.0
    positive_vals = heuristic[heuristic > 0]
    if len(positive_vals) > 0:
        med = np.median(positive_vals)
        if med > 0:
            heuristic = heuristic / med
    
    # Final safety
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Ensure diagonal is 0 (self-loops should not be attractive)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
