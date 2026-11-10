"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # wall-clock safety margin before 150s kill
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Pick next customer or return depot (0) to close route.
    
    Uses savings-based scoring with full-route simulation to minimize
    projected total route length.
    """
    unvisited_nodes = np.asarray(unvisited_nodes)
    if len(unvisited_nodes) == 0:
        return 0

    # If no feasible node fits in remaining capacity, close route
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]

    # Deadline guard: if too much time elapsed, fall back to nearest feasible
    if _elapsed() > _DEADLINE:
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    n = len(feasible_nodes)

    # Compute savings for each candidate:
    # savings(j) = dist(current, depot) + dist(depot, j) - dist(current, j)
    # High savings means merging current->j into this route saves more distance
    dist_current_depot = distance_matrix[current_node, depot]
    dist_depot_candidates = distance_matrix[depot, feasible_nodes]
    dist_current_candidates = distance_matrix[current_node, feasible_nodes]

    savings = dist_current_depot + dist_depot_candidates - dist_current_candidates

    # Full-route simulation: for each candidate, simulate the rest of the route
    # using a greedy savings+nearest hybrid, and compute total route length.
    # Vectorized where possible for speed.

    # For small n (typical in later stages), simulate each candidate's full route
    # For large n, use a faster approximate scoring

    if n <= 30:
        # Full simulation for each candidate
        best_score = np.inf
        best_node = feasible_nodes[0]

        for idx in range(n):
            cand = feasible_nodes[idx]
            # Simulate route: start at current, go to cand, then greedily pick
            # remaining feasible nodes using savings-based scoring
            sim_remaining_cap = rest_capacity - demands[cand]
            sim_visited = {int(cand)}
            sim_remaining = set(int(x) for x in unvisited_nodes if int(x) != int(cand))
            sim_current = int(cand)
            sim_total_dist = distance_matrix[current_node, cand]

            # Greedy completion: repeatedly pick node with best savings among feasible
            while sim_remaining:
                if _elapsed() > _DEADLINE:
                    break
                rem_arr = np.array(list(sim_remaining))
                cap_mask = demands[rem_arr] <= sim_remaining_cap
                if not np.any(cap_mask):
                    break
                rem_feasible = rem_arr[cap_mask]

                # Savings for each remaining feasible node
                dc_d = distance_matrix[sim_current, depot]
                d_d_rem = distance_matrix[depot, rem_feasible]
                d_cur_rem = distance_matrix[sim_current, rem_feasible]
                sav = dc_d + d_d_rem - d_cur_rem

                # Combine savings with distance (prefer high savings, low distance)
                # Normalize: use savings as primary, distance as tiebreaker
                # Scale distance to be comparable to savings
                dist_scale = np.max(d_cur_rem) if np.max(d_cur_rem) > 0 else 1.0
                score = sav - 0.5 * (d_cur_rem / dist_scale) * np.max(sav)

                next_idx = np.argmax(score)
                next_node = rem_feasible[next_idx]
                sim_total_dist += distance_matrix[sim_current, next_node]
                sim_remaining_cap -= demands[next_node]
                sim_remaining.discard(int(next_node))
                sim_current = int(next_node)

            # Close route: go back to depot
            sim_total_dist += distance_matrix[sim_current, depot]

            # Capacity-tightness penalty: if we're leaving many demands unvisited
            # due to capacity, penalize this choice
            unvisited_after = sim_remaining
            if unvisited_after:
                unvisited_arr = np.array(list(unvisited_after))
                total_unvisited_demand = np.sum(demands[unvisited_arr])
                # Penalty proportional to stranded demand
                penalty = 0.1 * total_unvisited_demand * np.mean(dist_current_depot)
            else:
                penalty = 0.0

            total_score = sim_total_dist + penalty

            if total_score < best_score:
                best_score = total_score
                best_node = cand

        return int(best_node)
    else:
        # Large n: use approximate scoring without full simulation
        # Score = -savings + distance + capacity_tightness_penalty
        # Higher savings is better (subtract it), lower distance is better

        # Capacity tightness: ratio of remaining capacity to remaining total demand
        total_remaining_demand = np.sum(demands[unvisited_nodes])
        if total_remaining_demand > 0:
            tightness = rest_capacity / total_remaining_demand
        else:
            tightness = 1.0

        # Penalty for picking a node when capacity is tight
        # If tightness is low, we should prefer nodes with lower demand (to save capacity)
        if tightness < 0.5:
            demand_penalty = 0.3 * demands[feasible_nodes] * dist_current_depot
        else:
            demand_penalty = np.zeros(n)

        # Composite score: lower is better
        # -savings (we want high savings), + distance, + demand penalty
        dist_scale = np.max(dist_current_candidates) if np.max(dist_current_candidates) > 0 else 1.0
        sav_scale = np.max(savings) if np.max(savings) > 0 else 1.0

        score = -savings / sav_scale + dist_current_candidates / dist_scale + demand_penalty / (dist_scale + 1e-9)

        best_idx = np.argmin(score)
        return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
