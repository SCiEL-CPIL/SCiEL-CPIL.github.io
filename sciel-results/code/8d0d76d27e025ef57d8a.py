"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return depot

    # If no feasible nodes, return to depot
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited_nodes])
    if not np.any(feasible_mask):
        return depot

    feasible_indices = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_indices)

    if n_feasible == 0:
        return depot

    # Compute distances from current node to each feasible candidate
    dists = distance_matrix[current_node, feasible_indices]

    # Compute distance from each feasible candidate to depot (return cost proxy)
    dists_to_depot = distance_matrix[depot, feasible_indices]

    # Capacity-aware: if remaining capacity is tight, prefer lower demand nodes
    total_remaining_demand = np.sum(demands[unvisited_nodes])
    # Average remaining demand per node
    avg_remaining_demand = total_remaining_demand / n_unvisited if n_unvisited > 0 else 1.0

    # If rest_capacity is less than 2x average demand, we're getting tight
    # In that case, strongly prefer lower-demand nodes
    capacity_tightness = max(0.0, 1.0 - rest_capacity / (2.0 * avg_remaining_demand))
    # capacity_tightness in [0, 1], higher means more urgent to save capacity

    # Score: immediate distance + alpha * return cost + beta * demand penalty
    # alpha: weight on return-to-depot cost (lookahead)
    # beta: weight on demand (when capacity is tight)
    alpha = 0.5
    beta = capacity_tightness * 0.3

    # Demand penalty: prefer lower demand when tight
    cand_demands = demands[feasible_indices]
    demand_penalty = beta * cand_demands

    # Outlier penalty: penalize distances significantly above mean (from observed improvement)
    if len(dists) > 2:
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        if std_d > 1e-9:
            # Penalize distances above mean by a fraction of the std
            outlier_penalty = 0.2 * np.maximum(0, dists - mean_d) / std_d
        else:
            outlier_penalty = np.zeros(len(dists))
    else:
        outlier_penalty = np.zeros(len(dists))

    scores = dists + alpha * dists_to_depot + demand_penalty + outlier_penalty

    best_idx = np.argmin(scores)
    return int(feasible_indices[best_idx])
# EVOLVE-BLOCK-END
