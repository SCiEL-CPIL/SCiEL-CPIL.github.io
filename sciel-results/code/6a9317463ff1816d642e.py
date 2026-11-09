"""CVRP heuristic: density-weighted centroid scoring with 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible, dtype=np.int64)
    n = len(feasible_arr)
    
    # Key distances
    dists_to_current = dm[current_node, feasible_arr]
    dists_to_depot = dm[feasible_arr, depot]
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)].copy()
    
    # 2-step lookahead: for each candidate i, estimate marginal route cost
    if n > 1:
        cost_ij_jd = sub_dm + dists_to_depot[None, :]
        np.fill_diagonal(cost_ij_jd, np.inf)
        best_next_cost = np.min(cost_ij_jd, axis=1)
    else:
        best_next_cost = np.array([0.0])
    
    # Density-weighted centroid: weight nodes by demand (heavier nodes matter more)
    demands_arr = np.array([demands[i] for i in feasible], dtype=np.float64)
    total_demand = np.sum(demands_arr)
    
    if total_demand > 0:
        weights = demands_arr / total_demand
    else:
        weights = np.ones(n) / n
    
    # Compute density-weighted centroid in distance space
    # We use distances to depot and between nodes as a proxy for spatial position
    # The centroid is a weighted average of "positions" represented by distance vectors
    
    # For each feasible node i, its "position" can be approximated by:
    # - distance to depot: dists_to_depot[i]
    # - distances to other nodes: sub_dm[i, :]
    
    # A simpler approach: use the distance to depot as a 1D proxy
    # and the average distance to other nodes as a second component
    
    # Weighted average distance to depot (centroid's "depot distance")
    centroid_depot_dist = np.sum(weights * dists_to_depot)
    
    # For each node i, compute how "central" it is relative to the centroid
    # Deviation from centroid in depot-distance space
    depot_deviation = np.abs(dists_to_depot - centroid_depot_dist)
    
    # Also consider how well-connected each node is to the cluster
    # Fill diagonal with 0 for this computation
    sub_dm_for_avg = sub_dm.copy()
    np.fill_diagonal(sub_dm_for_avg, 0)
    avg_dist_to_others = np.mean(sub_dm_for_avg, axis=1)
    
    # Nodes with lower average distance to others are more "central"
    max_avg_dist = np.max(avg_dist_to_others) if np.max(avg_dist_to_others) > 0 else 1.0
    centrality_score = 1.0 - (avg_dist_to_others / max_avg_dist)
    
    # Normalize depot deviation
    max_dev = np.max(depot_deviation) if np.max(depot_deviation) > 0 else 1.0
    norm_deviation = depot_deviation / max_dev
    
    # Savings-based component (from current program)
    savings = dists_to_depot[:, None] + dists_to_depot[None, :] - sub_dm
    avg_savings = np.mean(savings, axis=1)
    max_savings = np.max(avg_savings) if np.max(avg_savings) > 0 else 1.0
    norm_savings = avg_savings / max_savings
    
    # Composite score:
    # 1. Direct route cost: d(current,i) + best_next_cost[i]
    # 2. Centroid deviation penalty: prefer nodes near the density-weighted centroid
    # 3. Savings bonus: prefer nodes with high savings potential
    # 4. Centrality bonus: prefer well-connected nodes
    
    # Normalize the direct cost
    direct_cost = dists_to_current + best_next_cost
    max_direct = np.max(direct_cost) if np.max(direct_cost) > 0 else 1.0
    norm_direct = direct_cost / max_direct
    
    # Adaptive weights based on number of remaining nodes
    if n > 30:
        w_direct = 0.5
        w_centroid = 0.2
        w_savings = 0.2
        w_centrality = 0.1
    elif n > 15:
        w_direct = 0.55
        w_centroid = 0.2
        w_savings = 0.15
        w_centrality = 0.1
    elif n > 8:
        w_direct = 0.6
        w_centroid = 0.15
        w_savings = 0.15
        w_centrality = 0.1
    else:
        w_direct = 0.7
        w_centroid = 0.1
        w_savings = 0.1
        w_centrality = 0.1
    
    scores = (w_direct * norm_direct + 
              w_centroid * norm_deviation - 
              w_savings * norm_savings - 
              w_centrality * centrality_score)
    
    # Capacity-aware adjustment: when capacity is tight, prefer heavier nodes
    if total_demand > 0:
        util = rest_capacity / total_demand
    else:
        util = 1.0
    
    if util < 0.4:
        max_d = np.max(demands_arr) if np.max(demands_arr) > 0 else 1.0
        scores -= 0.1 * (demands_arr / max_d)
    
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
