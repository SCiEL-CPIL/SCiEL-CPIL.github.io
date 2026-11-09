"""CVRP heuristic: structural change to capacity-aware cluster + nearest-neighbor."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 150s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Structural change: instead of savings+lookahead per-step, use a 
    capacity-aware nearest neighbor with 'return pressure'.
    
    Key idea: as remaining capacity drops, we should prefer nodes that are
    close to the depot (to allow a clean return) rather than just closest
    to current. This creates natural clusters.
    
    Score = d(current, i) + alpha * d(i, depot) where alpha increases as
    rest_capacity decreases relative to the average demand.
    """
    if len(unvisited_nodes) == 0:
        return 0

    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if np.any(feasible_mask):
            feasible_nodes = unvisited_nodes[feasible_mask]
            return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
        return 0

    # Filter to feasible nodes
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distances
    dists = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    dist_to_depot = distance_matrix[feasible_nodes, depot]  # (n_feasible,)

    # Capacity pressure: how much capacity is left relative to what we need
    # If rest_capacity is small relative to total remaining demand, we should
    # be heading back to depot soon
    total_remaining_demand = np.sum(demands[unvisited_nodes])
    if total_remaining_demand > 0:
        # Ratio of remaining capacity to remaining demand (normalized)
        # When this ratio is low, we're capacity-constrained and should return
        capacity_ratio = rest_capacity / (total_remaining_demand + 1e-9)
    else:
        capacity_ratio = 1.0

    # Alpha: weight on distance to depot
    # When capacity is tight (low ratio), alpha is high -> prefer depot-close nodes
    # When capacity is ample (high ratio), alpha is low -> prefer nearest neighbor
    # Use a smooth transition
    # capacity_ratio is roughly in [0, 1] range for typical instances
    # Map to alpha in [0.1, 1.5]
    alpha = 0.1 + 1.4 * (1.0 - min(capacity_ratio, 1.0))

    # Score: lower is better
    scores = dists + alpha * dist_to_depot

    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
