"""Model-and-optimize heuristic for Capacitated VRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic combining Clarke-Wright savings with 2-step lookahead.
    
    Strategy:
    1. Use savings-based scoring: prefer nodes that reduce total route length
       compared to going back to depot and visiting separately.
    2. Add a 2-step lookahead to estimate the cost of the next best move after
       visiting each candidate.
    3. Apply a small capacity-aware penalty to avoid premature route closure.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: nearest feasible node
        unvisited_arr = np.asarray(unvisited_nodes)
        dm_arr = np.asarray(distance_matrix)
        dems_arr = np.asarray(demands)
        feasible = unvisited_arr[dems_arr[unvisited_arr] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(dm_arr[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.asarray(unvisited_nodes)
    dm = np.asarray(distance_matrix)
    dems = np.asarray(demands)

    # Filter to feasible customers
    feasible = unvisited[dems[unvisited] <= rest_capacity]
    if len(feasible) == 0:
        return 0

    n_feasible = len(feasible)

    if n_feasible == 1:
        return int(feasible[0])

    # Distances from current node to each feasible
    dist_cur = dm[current_node, feasible]

    # Distances back to depot
    dist_depot = dm[feasible, depot]

    # Demand of each feasible customer
    reqs = dems[feasible]

    # Distance from current node to depot (baseline for savings)
    d_cur_depot = dm[current_node, depot]

    # Clarke-Wright savings: how much do we save by visiting i now vs. going to depot first?
    # If we go current -> depot -> i: cost = d_cur_depot + dm[depot, i]
    # If we go current -> i -> depot: cost = dist_cur[i] + dist_depot[i]
    # Savings = (d_cur_depot + dm[depot, i]) - (dist_cur[i] + dist_depot[i])
    # We want to MAXIMIZE savings (i.e., minimize negative savings)
    savings = d_cur_depot + dm[depot, feasible] - dist_cur - dist_depot

    # 2-step lookahead: for each candidate i, estimate the best next move
    lookahead_costs = np.zeros(n_feasible)

    for i in range(n_feasible):
        node_i = feasible[i]
        remaining_cap = rest_capacity - reqs[i]

        # Options: go to depot, or go to another feasible node that fits
        best_next_cost = dm[node_i, depot]  # default: return to depot

        if remaining_cap > 0:
            # Find other feasible nodes that would fit after visiting i
            other_feasible = unvisited[dems[unvisited] <= remaining_cap]
            mask = other_feasible != node_i
            others = other_feasible[mask]

            if len(others) > 0:
                dists_i_j = dm[node_i, others]
                dists_j_depot = dm[others, depot]
                next_costs = dists_i_j + dists_j_depot
                best_next_cost = min(best_next_cost, np.min(next_costs))

        lookahead_costs[i] = dist_cur[i] + best_next_cost

    # Combine savings and lookahead:
    # We want to minimize total cost, so we score by:
    #   primary: lookahead cost (2-step estimate)
    #   secondary: negative savings (prefer higher savings)
    #   tertiary: small capacity penalty to avoid overfilling

    # Normalize savings to be comparable with distances
    # Savings can be positive or negative; we subtract it (so higher savings = lower score)
    score = lookahead_costs - 0.5 * savings

    # Small capacity-aware penalty: discourage taking a node that uses a large fraction
    # of remaining capacity, as this may force an early return
    if rest_capacity > 0:
        cap_frac = reqs / rest_capacity
        score += 0.05 * cap_frac * dist_depot

    best_idx = int(np.argmin(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
