"""Guided local search penalty rule for TSP with adaptive edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply adaptive penalties to edges in the current tour based on their usage count.
    Edges used more frequently get higher penalties to encourage exploration.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Extract the edges of the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))

    # Get the usage counts for these edges
    edge_counts = np.array([edge_n_used[u, v] for u, v in tour_edges], dtype=np.float64)

    # Compute a penalty multiplier for each edge in the tour
    # Use a smooth, adaptive penalty that grows with usage
    # Base penalty rate: 0.15 (moderate)
    # Normalize by max usage to keep penalties bounded
    max_usage = np.max(edge_counts) if np.max(edge_counts) > 0 else 1.0
    norm_counts = edge_counts / (1.0 + max_usage)

    # Penalty multiplier: 1 + base_rate * normalized_count
    # This ensures edges with higher usage get penalized more
    base_rate = 0.15
    multipliers = 1.0 + base_rate * norm_counts

    # Build the penalized distance matrix
    penalized = edge_distance.copy()
    for (u, v), mult in zip(tour_edges, multipliers):
        penalized[u, v] *= mult
        penalized[v, u] *= mult

    # Ensure symmetry and finiteness
    penalized = np.maximum(penalized, 0.0)
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)

    return penalized
# EVOLVE-BLOCK-END
