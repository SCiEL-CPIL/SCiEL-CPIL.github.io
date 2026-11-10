"""Improved heuristic for CVRP using savings-based selection with early termination."""
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # Convert to arrays for speed
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    if len(unvisited_arr) == 0:
        return 0

    demands_arr = np.array(demands, dtype=float)
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity

    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_arr[feasible_mask]

    # Distances
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[feasible_nodes, depot]

    # If we are at the depot, start with nearest neighbor
    if current_node == depot:
        best_idx = np.argmin(dists_current)
        return int(feasible_nodes[best_idx])

    # Clarke-Wright savings: minimize d(i,j) - d(0,j)
    savings_cost = dists_current - dists_depot

    # Early termination: if the best extension is not better than returning to depot,
    # close the route now.
    # Cost to return to depot: d(i, 0)
    # Cost to extend to j and later return: d(i,j) + d(j,0)
    # We should extend only if d(i,j) + d(j,0) < d(i,0) + d(0,j)
    # i.e., d(i,j) - d(0,j) < d(i,0)
    min_savings_cost = np.min(savings_cost)
    dist_i_depot = distance_matrix[current_node, depot]

    if min_savings_cost >= dist_i_depot:
        return 0

    # Add a small demand-based urgency component to prioritize high-demand nodes
    # This helps fill capacity and reduce the number of trips
    feas_demands = demands_arr[feasible_nodes]
    max_demand = feas_demands.max() if len(feas_demands) > 0 else 1.0
    if max_demand > 0:
        demand_urgency = 0.15 * (feas_demands / max_demand)
        final_scores = savings_cost - demand_urgency
    else:
        final_scores = savings_cost

    best_idx = np.argmin(final_scores)

    return int(feasible_nodes[best_idx])
