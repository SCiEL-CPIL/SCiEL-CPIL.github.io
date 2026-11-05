"""Improved CVRP heuristic: multi-layered savings with capacity-aware clustering."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Convert to numpy arrays for fast operations
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Feasibility: demand must fit in remaining capacity
    demands_unvisited = demands[unvisited_arr]
    feas_mask = demands_unvisited <= rest_capacity
    
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)
    
    if k == 1:
        return int(feas[0])

    # Distances from current node to all feasible candidates
    d_cur = distance_matrix[current_node, feas]
    d_cur_depot = distance_matrix[current_node, depot]
    d_depot_feas = distance_matrix[depot, feas]
    
    # Clarke-Wright savings: how much we save by visiting feas[i] before going back to depot
    savings = d_cur_depot + d_depot_feas - d_cur

    # Two-step lookahead: for each candidate i, estimate the cost of the NEXT edge
    # by finding the nearest other feasible node j (excluding i itself)
    # This encourages picking nodes that keep the route compact
    if k > 1:
        feas_dm = distance_matrix[np.ix_(feas, feas)].copy()
        np.fill_diagonal(feas_dm, np.inf)
        nn_dist = feas_dm.min(axis=1)  # nearest neighbor distance for each feasible node
        
        # Second-order: after going to i, what's the average distance to remaining nodes?
        # Use a soft approximation: mean distance from i to all other feasible nodes
        mean_dist_from_i = feas_dm.sum(axis=1) / (k - 1)
    else:
        nn_dist = np.zeros(1)
        mean_dist_from_i = np.zeros(1)

    # Capacity pressure: as rest_capacity shrinks relative to total remaining demand,
    # we should prefer smaller-demand nodes to avoid getting stuck
    total_remaining_demand = demands_unvisited[feas_mask].sum()
    if total_remaining_demand > 0:
        capacity_pressure = 1.0 - (rest_capacity / total_remaining_demand)
        capacity_pressure = np.clip(capacity_pressure, 0.0, 1.0)
    else:
        capacity_pressure = 0.0

    # Normalize demands for weighting
    max_demand = demands_unvisited[feas_mask].max()
    if max_demand > 0:
        norm_demand = demands_unvisited[feas_mask] / max_demand
    else:
        norm_demand = np.zeros(k)

    # Compute regret: penalize edges that are much longer than average
    mean_d_cur = np.mean(d_cur)
    regret = np.maximum(0, d_cur - mean_d_cur)

    # Multi-component score:
    # 1. Primary: distance from current (direct cost)
    # 2. Savings: reward compact routes (higher savings = better)
    # 3. Lookahead: penalize isolated nodes (high nn_dist)
    # 4. Second-order: penalize nodes far from the rest of the cluster
    # 5. Capacity-aware demand penalty: when capacity is tight, prefer low-demand nodes
    # 6. Regret: penalize outlier-long edges
    
    alpha = 1.0      # weight for direct distance
    beta = 0.85      # weight for savings (higher = more compact routes)
    gamma = 0.25     # weight for nearest-neighbor lookahead
    delta = 0.15     # weight for mean-distance lookahead
    epsilon = 0.2    # weight for regret
    zeta = capacity_pressure * 0.3  # dynamic capacity-aware demand penalty
    
    scores = (
        alpha * d_cur
        - beta * savings
        + gamma * nn_dist
        + delta * mean_dist_from_i
        + epsilon * regret
        + zeta * norm_demand * d_cur_depot
    )

    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
