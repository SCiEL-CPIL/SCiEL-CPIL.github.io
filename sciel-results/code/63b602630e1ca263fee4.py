"""Guided local search perturbation for TSP with 2-opt move regret model."""
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search using a 2-opt move regret model.
    
    For each edge in the current tour, compute the minimal length increase (regret)
    required to remove it via a single 2-opt swap. Then penalize edges with high regret
    (i.e., edges that are 'hard' to remove) less, and edges with low regret (easy to remove)
    more, encouraging the modified 2-opt to break them.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    N = len(local_opt_tour)
    if N < 3:
        return edge_distance.copy()
    
    # Extract tour edges: (tour[i], tour[i+1]) for i in 0..N-2, and (tour[N-1], tour[0])
    tour = local_opt_tour
    tour_edges = []
    for i in range(N):
        u = tour[i]
        v = tour[(i + 1) % N]
        tour_edges.append((u, v))
    
    # Compute regret for each tour edge: minimal length increase to remove it via 2-opt
    # A 2-opt move removing edge (u,v) and edge (a,b) replaces them with (u,a) and (v,b)
    # or (u,b) and (v,a). The length change is:
    #   delta = d(u,a) + d(v,b) - d(u,v) - d(a,b)
    # or
    #   delta = d(u,b) + d(v,a) - d(u,v) - d(a,b)
    # We want the minimal positive delta (or 0 if negative, meaning it would improve)
    
    # For efficiency, precompute tour edge indices
    # Map from (u,v) to index in tour_edges for quick lookup
    # Actually, let's compute regret for each tour edge by considering all other edges
    
    # To keep this tractable, we'll use a simplified approach:
    # For each tour edge (u,v), find the best 2-opt move that removes it.
    # We consider all pairs of non-adjacent edges in the tour.
    
    # Precompute: for each node, its position in the tour
    pos_in_tour = np.zeros(N, dtype=int)
    for i, node in enumerate(tour):
        pos_in_tour[node] = i
    
    # Compute regret for each tour edge
    # regret[i] = minimal length increase to remove tour edge i
    regrets = np.zeros(N)
    
    # For each tour edge i, consider removing it
    for i in range(N):
        u, v = tour_edges[i]
        best_delta = np.inf
        
        # Consider all other edges j (not adjacent to i)
        for j in range(N):
            if j == i or j == (i - 1) % N or j == (i + 1) % N:
                continue
            
            a, b = tour_edges[j]
            # Two ways to reconnect:
            # Option 1: (u,a) and (v,b)
            delta1 = edge_distance[u, a] + edge_distance[v, b] - edge_distance[u, v] - edge_distance[a, b]
            # Option 2: (u,b) and (v,a)
            delta2 = edge_distance[u, b] + edge_distance[v, a] - edge_distance[u, v] - edge_distance[a, b]
            
            delta = min(delta1, delta2)
            if delta < best_delta:
                best_delta = delta
        
        # If best_delta is negative, it means removing this edge would improve the tour,
        # so regret is 0 (no penalty needed, it's already "easy" to remove)
        regrets[i] = max(0.0, best_delta) if best_delta < np.inf else 0.0
    
    # Now set up the penalty:
    # We want to penalize edges with LOW regret (easy to remove) MORE,
    # and edges with HIGH regret (hard to remove) LESS.
    # So penalty_factor * (max_regret - regret[i])
    
    max_regret = np.max(regrets) if N > 0 else 0.0
    if max_regret < 1e-12:
        max_regret = 1.0
    
    # Create perturbed distance matrix
    perturbed = edge_distance.copy()
    
    # Penalty factor: scale relative to typical edge lengths
    # Use a moderate penalty factor
    penalty_factor = 0.5
    
    for i in range(N):
        u, v = tour_edges[i]
        regret = regrets[i]
        # Penalty is higher for edges with lower regret
        penalty = penalty_factor * (max_regret - regret) / max_regret
        # Add penalty to the edge distance
        perturbed[u, v] += penalty
        perturbed[v, u] += penalty
    
    # Also incorporate usage-based penalty for edges not in the current tour
    # to repel from historical local optima
    if edge_n_used.max() > 0:
        usage_penalty = 0.1 * edge_n_used / (1.0 + edge_n_used.max())
        # Only apply to edges not already penalized above (or add to all)
        # Let's add a small usage-based penalty to all edges
        perturbed += usage_penalty
    
    # Ensure symmetry and finite values
    perturbed = (perturbed + perturbed.T) / 2.0
    perturbed = np.where(np.isfinite(perturbed), perturbed, 0.0)
    
    return perturbed
