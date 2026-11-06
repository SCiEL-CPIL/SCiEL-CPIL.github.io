"""Model-and-optimize heuristic for Capacitated VRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic combining Clarke-Wright savings, 2-step lookahead, regret, and spatial density.
    
    Strategy:
    1. Use savings-based scoring: prefer nodes that reduce total route length.
    2. Add a 2-step lookahead to estimate the cost of the next best move.
    3. Apply regret-based selection: prioritize nodes that are significantly better than the second-best alternative.
    4. Apply a spatial density bonus: prioritize nodes in dense clusters (low distance to neighbors) to localize demand.
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
    # Savings = (d_cur_depot + dm[depot, i]) - (dist_cur[i] + dist_depot[i])
    # We want to MAXIMIZE savings
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
                # We just need the distance to the next node for the 2-step cost estimate
                # Cost = dist_cur[i] + dist(i, next_best)
                # We estimate next_best as the one with min dist(i, j)
                # Note: We don't add dist(j, depot) here because that's part of the "next" decision,
                # but for a 2-step lookahead, we often just look at the immediate connection cost.
                # However, the previous version added dists_j_depot. Let's stick to a consistent definition.
                # Previous: best_next_cost = min(dist(i, depot), min_j(dist(i, j) + dist(j, depot)))
                # This is a 3-step lookahead effectively (i -> j -> depot).
                # Let's keep it as it was, it worked.
                dists_j_depot = dm[others, depot]
                next_costs = dists_i_j + dists_j_depot
                best_next_cost = min(best_next_cost, np.min(next_costs))

        lookahead_costs[i] = dist_cur[i] + best_next_cost

    # Regret-based scoring:
    # For each candidate i, regret is the difference between the best score (excluding i) and the score of i.
    # We want to MAXIMIZE regret (i.e., pick the node that is uniquely best).
    # Score to minimize is -savings + lookahead_cost. Let's define a "raw cost" for regret.
    # Lower raw cost is better.
    # raw_cost = lookahead_costs - savings
    raw_cost = lookahead_costs - savings
    
    # Sort raw_cost to find best and second best
    sorted_indices = np.argsort(raw_cost)
    best_idx_global = sorted_indices[0]
    best_cost_global = raw_cost[best_idx_global]
    
    if n_feasible > 1:
        second_best_idx_global = sorted_indices[1]
        second_best_cost_global = raw_cost[second_best_idx_global]
    else:
        second_best_cost_global = np.inf

    regret_scores = np.zeros(n_feasible)
    for i in range(n_feasible):
        if i == best_idx_global:
            # If i is the global best, the best alternative is the second best global
            regret_scores[i] = best_cost_global - second_best_cost_global
        else:
            # If i is not the global best, the best alternative is the global best
            regret_scores[i] = best_cost_global - raw_cost[i]
            
    # Spatial Density / Clustering Bonus:
    # Estimate how "clustered" each node is. A node in a dense area has low distances to its neighbors.
    # This encourages filling capacity in a local region.
    # We approximate Voronoi density by summing distances to the 3 nearest neighbors among all unvisited nodes.
    # Lower sum = denser cluster = better.
    # We add this as a penalty (we want to minimize it, but we want to *prefer* low values, so we subtract a negative or add a positive penalty if high?).
    # Let's define density_score[i] = sum of distances to k nearest neighbors.
    # We want to MINIMIZE total route length.
    # If we pick a node in a dense cluster, we likely pick its neighbors next, which are close.
    # So, we should PREFER nodes with LOW density_score (low sum of neighbor distances).
    # Therefore, we subtract a weighted density_score from our total score?
    # Or add a penalty for HIGH density_score?
    # Let's add: score += 0.1 * density_score[i]
    # Wait, if density_score is small (good), we want score to be small.
    # So score += 0.1 * density_score is correct.
    
    density_scores = np.zeros(n_feasible)
    # To speed this up, we only do this if n_feasible is not too large, or just do it for all since n<=50.
    for i in range(n_feasible):
        node_i = feasible[i]
        # Distances to all other unvisited nodes
        dists_to_others = dm[node_i, unvisited]
        # Exclude self
        dists_to_others[unvisited == node_i] = np.inf
        # Get 3 nearest
        k = min(3, len(unvisited) - 1)
        if k > 0:
            nearest_dists = np.sort(dists_to_others)[:k]
            density_scores[i] = np.sum(nearest_dists)
        else:
            density_scores[i] = 0.0

    # Combine scores:
    # We want to minimize the final score.
    # Primary: lookahead_costs (estimated future cost)
    # Secondary: -savings (we want high savings, so subtract savings)
    # Tertiary: -regret (we want high regret, so subtract regret)
    # Quaternary: density_scores (we want low density scores, so add them)
    
    # Weights:
    # Lookahead is the most direct estimate of cost.
    # Savings is a good proxy for immediate edge quality.
    # Regret helps break ties or identify unique bests.
    # Density helps with global structure.
    
    score = 1.0 * lookahead_costs - 1.0 * savings - 0.5 * regret_scores + 0.1 * density_scores

    # Small capacity-aware penalty: discourage taking a node that uses a large fraction
    # of remaining capacity, as this may force an early return
    if rest_capacity > 0:
        cap_frac = reqs / rest_capacity
        score += 0.05 * cap_frac * dist_depot

    best_idx = int(np.argmin(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
