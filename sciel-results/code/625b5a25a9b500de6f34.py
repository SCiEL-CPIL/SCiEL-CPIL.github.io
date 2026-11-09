"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Savings + continuation-cost heuristic with time guard."""
    if len(unvisited_nodes) == 0:
        return 0

    # Time guard: if we're running out of time, fall back to nearest feasible
    if time.time() - _START_TIME > _TIME_LIMIT:
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if np.any(feasible_mask):
            feasible_nodes = unvisited_nodes[feasible_mask]
            return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
        return 0

    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distance from current node to each feasible candidate
    dists = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    dist_to_depot = distance_matrix[feasible_nodes, depot]  # (n_feasible,)
    dist_depot_to_curr = distance_matrix[depot, current_node]  # scalar

    # Savings of visiting candidate i instead of going back to depot:
    # saving = d(depot, current) + d(i, depot) - d(current, i)
    savings = dist_depot_to_curr + dist_to_depot - dists

    # Continuation cost: after visiting candidate i, estimate min distance to next feasible node or depot
    # This encourages clustered routes by penalizing nodes that leave you far from remaining customers
    if n_feasible > 1:
        # For each feasible node i, compute min distance from i to any other feasible node or to depot
        # distance_matrix[feasible_nodes, feasible_nodes] gives pairwise distances
        pair_dists = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]  # (n_feasible, n_feasible)
        # Set diagonal to inf so we don't count self-distance
        np.fill_diagonal(pair_dists, np.inf)
        min_dist_to_next = np.min(pair_dists, axis=1)  # (n_feasible,)
        # Also consider distance to depot as an option for "next"
        continuation_cost = np.minimum(min_dist_to_next, dist_to_depot)
    else:
        continuation_cost = dist_to_depot.copy()

    # Normalize components
    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    max_saving = np.max(savings) if np.max(savings) > 0 else 1.0
    max_cont = np.max(continuation_cost) if np.max(continuation_cost) > 0 else 1.0

    # Weighted score: distance + continuation cost, minus savings bonus
    # Lower score is better
    alpha_saving = 0.25   # weight for savings (prefer higher savings)
    alpha_cont = 0.35     # weight for continuation cost (prefer lower)
    
    scores = (dists / max_dist + alpha_cont * continuation_cost / max_cont 
              - alpha_saving * savings / max_saving)

    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
