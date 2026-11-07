"""Guided local search penalty for TSP – refined within the multiplicative-scaling family."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search.
    
    Strategy:
      - Penalize edges that belong to the current local-optimal tour (the "sticking" edges).
      - Scale the penalty by a smooth, bounded function of the edge's cumulative usage count.
      - Use a multiplicative factor so that relative distances are preserved for non-tour edges.
      - Add a small global offset to avoid zero/negative entries and keep the matrix finite.
    
    The key idea: edges in the current tour that have been used many times get a larger
    multiplicative penalty, encouraging the 2-opt re-optimization to break them.
    """
    n = edge_distance.shape[0]
    deadline = time.time() + 290.0  # safety margin within 330s wall-clock
    
    # Work on a copy
    perturbed = edge_distance.astype(np.float64).copy()
    
    # Extract edges of the current tour
    # local_opt_tour is a sequence of node indices forming a closed tour
    tour = np.asarray(local_opt_tour, dtype=np.intp)
    m = len(tour)
    
    if m < 2:
        return perturbed
    
    # For each consecutive pair in the tour (including wrap-around), get the usage count
    tour_edges = []
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        tour_edges.append((u, v))
    
    # Compute the penalty factor for each tour edge based on its usage count
    # Use a smooth saturating function: penalty_weight = 1 + alpha * (count / (1 + count))
    # This is bounded in [1, 1+alpha) and grows with usage.
    alpha = 1.5  # max additional multiplicative factor for heavily-used tour edges
    
    # Get usage counts for tour edges
    counts = np.array([edge_n_used[u, v] for u, v in tour_edges], dtype=np.float64)
    
    # Smooth bounded scaling: w = 1 + alpha * count / (1 + count)
    weights = 1.0 + alpha * counts / (1.0 + counts)
    
    # Apply multiplicative penalty only to tour edges
    for (u, v), w in zip(tour_edges, weights):
        perturbed[u, v] *= w
        perturbed[v, u] *= w
    
    # Additionally, apply a mild penalty to all edges with high usage, not just tour edges.
    # This helps escape from frequently-used substructures.
    # Use a smaller alpha for non-tour edges to avoid over-perturbation.
    alpha_global = 0.3
    # Only apply to edges where usage > 0 to save time
    mask = edge_n_used > 0
    if np.any(mask):
        # Compute global weights
        global_weights = 1.0 + alpha_global * edge_n_used[mask] / (1.0 + edge_n_used[mask])
        perturbed[mask] *= global_weights
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) * 0.5
    
    # Ensure diagonal is zero (self-loops)
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure all values are finite and positive (except diagonal)
    if not np.all(np.isfinite(perturbed)):
        perturbed = np.nan_to_num(perturbed, nan=1e6, posinf=1e6, neginf=1e6)
    
    # Check deadline
    if time.time() > deadline:
        return perturbed
    
    return perturbed
# EVOLVE-BLOCK-END
