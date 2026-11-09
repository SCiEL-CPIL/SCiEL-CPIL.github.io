"""Guided local search penalty rule for the TSP with topological disruption."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy: Topological Disruption
    1. Identify the 'skeleton' of the current local optimum: the k shortest edges 
       in the current tour. These edges are likely the most 'stable' and define 
       the local basin's structure.
    2. Apply a massive penalty (effectively infinity) to these specific edges.
       This forces 2-opt to break the tour and reassemble it, preventing it 
       from just swapping in a slightly longer edge.
    3. Apply a smaller, usage-based penalty to other current tour edges to 
       discourage immediate return to the same configuration.
    4. This shifts the search from 'refinement' to 'restructuring'.
    """
    # Check deadline: if we are too close to the limit, return a safe, 
    # minimally perturbed matrix to avoid timeout.
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    D = edge_distance.copy()

    # Extract edges of the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v, edge_distance[u, v]))
    
    if not tour_edges:
        return D

    # Sort edges by length to find the 'skeleton' (shortest edges)
    tour_edges.sort(key=lambda x: x[2])
    
    # Define how many shortest edges to disrupt. 
    # Disrupting ~10-20% of the tour forces significant restructuring.
    k = max(2, int(0.15 * n))
    skeleton_edges = tour_edges[:k]
    
    # Calculate a scale factor for the 'soft' penalty on remaining edges
    avg_edge_len = np.mean([e[2] for e in tour_edges])
    if not np.isfinite(avg_edge_len) or avg_edge_len <= 0:
        avg_edge_len = 1.0

    # 1. Topological Disruption: Set the shortest edges to a very high cost.
    # Using a large multiple of the max edge length ensures they are never chosen.
    max_edge_len = np.max(edge_distance)
    if not np.isfinite(max_edge_len) or max_edge_len <= 0:
        max_edge_len = 1.0
    
    disruption_cost = 10.0 * max_edge_len
    
    for u, v, _ in skeleton_edges:
        D[u, v] = disruption_cost
        D[v, u] = disruption_cost

    # 2. Soft Penalty on Remaining Current Tour Edges
    # Penalize the rest of the current tour edges based on usage count to 
    # discourage staying in the same basin, but less aggressively than the skeleton.
    # This helps if the disruption wasn't enough to fully escape.
    soft_penalty_scale = 0.2 * avg_edge_len
    
    for u, v, _ in tour_edges[k:]:
        usage = edge_n_used[u, v]
        # Add a penalty proportional to usage. 
        # Since these are in the current tour, usage is at least 1.
        penalty = soft_penalty_scale * usage
        D[u, v] += penalty
        D[v, u] += penalty

    # 3. Optional: Light penalty on high-usage non-tour edges to avoid known bad basins
    # We only do this if we have time, to keep it fast.
    # For now, we focus on the disruption.
    
    # Ensure diagonal is 0
    np.fill_diagonal(D, 0.0)

    # Ensure symmetry and finiteness
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
