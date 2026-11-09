"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_MODULE_START = time.time()
TIME_LIMIT = 115.0  # seconds; safety margin below the 120s/150s limits


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP next-node selection.
    
    Strategy: Among feasible unvisited customers (demand fits rest_capacity),
    pick the one minimizing a composite score:
      - Primary: distance from current node to candidate (route continuity)
      - Secondary: distance from candidate to depot (adaptive weight; higher
        when capacity is tight to avoid ending far from depot)
      - Tertiary: soft-min distance to other feasible nodes (encourages clustering)
    
    Uses a smooth surrogate (soft-min via log-sum-exp) for the nearest-other term
    to avoid discontinuities that can cause suboptimal local choices.
    
    If no feasible customer exists, return depot (0) to close the route.
    """
    # Deadline guard: if we're running long, just pick the nearest feasible node quickly
    elapsed = time.time() - _MODULE_START
    if elapsed > TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda u: distance_matrix[current_node, u]))

    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity feasibility
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]

    # Distances from each feasible node to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]

    # Compute soft-min distance from each feasible node to any OTHER feasible node.
    # This encourages clustering while using a smooth surrogate for the min operation.
    # sub_matrix[i, j] = distance from feasible_nodes[i] to feasible_nodes[j]
    sub_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    # Set diagonal to a large value so self is not considered
    np.fill_diagonal(sub_matrix, np.inf)

    # Soft-min: for each row i, soft_min_j(sub_matrix[i,j]) 
    # = -beta * log(sum_j exp(-sub_matrix[i,j] / beta))
    # As beta -> 0, this approaches the true min. We use beta=1.0 for smoothness.
    beta = 1.0
    # For numerical stability, subtract row max before exp
    row_max = np.max(-sub_matrix, axis=1, keepdims=True)
    # Where all entries are inf (shouldn't happen for n>1), handle gracefully
    row_max_safe = np.where(np.isinf(row_max), 0.0, row_max)
    exp_terms = np.exp((-sub_matrix - row_max_safe) / beta)
    # Where sub_matrix was inf, exp(-inf) = 0, which is fine
    log_sum = np.log(np.sum(exp_terms, axis=1) + 1e-300)
    soft_min_dist = -beta * (log_sum - row_max_safe.flatten())
    # Replace any non-finite values with a large penalty
    soft_min_dist = np.where(np.isfinite(soft_min_dist), soft_min_dist, 1e6)

    # Adaptive weight for depot distance:
    # When rest_capacity is small relative to remaining demand, we should prefer
    # nodes closer to depot to avoid ending the route far away.
    total_remaining_demand = np.sum(demands[unvisited_nodes])
    if total_remaining_demand > 0:
        cap_ratio = rest_capacity / total_remaining_demand
    else:
        cap_ratio = 1.0

    # w_depot ranges from ~0.1 (plenty of capacity) to ~0.6 (tight capacity)
    w_depot = 0.5 * (1.0 - cap_ratio) + 0.1
    # Weight for clustering term
    w_cluster = 0.3

    # Composite score: lower is better
    scores = dists_from_current + w_depot * dists_to_depot + w_cluster * soft_min_dist

    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
