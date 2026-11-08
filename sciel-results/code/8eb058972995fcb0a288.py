"""Guided local search penalty rule for the TSP with global usage-based penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - Apply a global additive penalty to ALL edges based on their usage count.
    - Unlike penalizing only current tour edges, this penalizes the entire 
      "history" of the search, making previously visited edges expensive 
      everywhere in the landscape.
    - This forces the local search to explore new, unused edges rather than 
      oscillating between a few known local optima.
    - The penalty is scaled by a factor of the average edge length to ensure 
      it is significant relative to the problem scale.
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

    # Calculate a scale factor based on the average edge length of the current tour.
    # This ensures the penalty is proportional to the problem's geometry.
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # If the tour length is zero or invalid, use a default scale
    if not np.isfinite(avg_edge_len) or avg_edge_len <= 0:
        avg_edge_len = 1.0

    # Define the penalty coefficient.
    # We want the penalty to grow with usage but not dominate immediately.
    # alpha = 0.1 * avg_edge_len means an edge used 10 times gets a penalty 
    # of 1.0 * avg_edge_len, which is substantial.
    alpha = 0.1 * avg_edge_len

    # Apply the penalty to all edges with non-zero usage.
    # We iterate over all pairs to apply the penalty where edge_n_used > 0.
    # For efficiency, we can use numpy operations, but edge_n_used is likely sparse 
    # or dense depending on the evaluator. Assuming dense for 100 nodes (10k entries) 
    # is fast enough.
    
    # Vectorized application:
    # penalty_matrix = alpha * edge_n_used
    # D += penalty_matrix
    
    # However, we must be careful not to penalize the diagonal (self-loops) 
    # if they exist, though TSP tours don't use them.
    # Also, we should ensure we don't penalize edges that are not in the graph 
    # (if any), but edge_distance is a full matrix.
    
    # Let's do it vectorized for speed.
    penalty = alpha * edge_n_used
    
    # Ensure diagonal is 0 penalty (just in case edge_n_used has diagonal values)
    np.fill_diagonal(penalty, 0.0)
    
    D += penalty

    # Ensure symmetry and finiteness
    # The addition preserves symmetry if input was symmetric and penalty is symmetric.
    # edge_n_used is symmetric, so penalty is symmetric.
    
    if not np.all(np.isfinite(D)):
        # Fallback to true distances if something went wrong
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
