"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    
    Uses a combination of:
    1. Global usage-based penalty (smooth, scaled by usage frequency)
    2. Topological disruption: penalize the k shortest edges in the current tour
       to force restructuring rather than minor swaps
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances (finite, same shape).
    """
    n = edge_distance.shape[0]
    
    # Start with a copy of the true distances
    perturbed = edge_distance.astype(np.float64).copy()
    
    # --- Component 1: Global usage-based penalty ---
    # Penalize edges proportionally to how often they've been used in local optima.
    # Use a smooth, bounded penalty to avoid extreme values.
    max_used = edge_n_used.max()
    if max_used > 0:
        # Smooth penalty: alpha * (usage / (usage + 1))
        # This gives penalty ~ alpha for heavily used edges, ~ 0 for unused
        alpha = 0.5  # Penalty strength
        usage_ratio = edge_n_used / (1.0 + edge_n_used)
        perturbed += alpha * usage_ratio
    
    # --- Component 2: Topological disruption ---
    # Identify the k shortest edges in the current tour and penalize them strongly.
    # This forces 2-opt to consider alternative connections.
    
    # Extract tour edges (including the closing edge)
    tour = local_opt_tour
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edges.append((u, v, edge_distance[u, v]))
    
    # Sort by distance and pick the k shortest
    tour_edges.sort(key=lambda x: x[2])
    k = max(1, int(0.1 * n))  # 10% of edges, at least 1
    
    # Penalize these k shortest edges with a large constant
    # Use a value that's large enough to discourage use but not infinite
    # to keep the matrix finite and well-conditioned
    max_dist = edge_distance.max()
    disruption_penalty = 2.0 * max_dist  # Twice the max distance
    
    for i in range(k):
        u, v, _ = tour_edges[i]
        perturbed[u, v] += disruption_penalty
        perturbed[v, u] += disruption_penalty
    
    # Ensure the diagonal stays zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure finiteness and non-negativity
    perturbed = np.where(np.isfinite(perturbed), perturbed, 1e10)
    perturbed = np.maximum(perturbed, 0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
