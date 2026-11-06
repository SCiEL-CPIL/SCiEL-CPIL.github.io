"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node to all feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Compute regret-based score:
    # For each feasible node i, regret = distance to nearest OTHER feasible node - distance to itself
    # But that's not quite right. Let's think about this differently.
    # 
    # Regret-based selection: prioritize nodes that are uniquely well-positioned.
    # A node has high regret if its connection cost is much lower than alternatives.
    # 
    # Alternative approach: score each node by a combination of:
    # 1. Distance from current node (lower is better)
    # 2. Regret: how much worse it would be to not visit this node now
    #    Regret can be approximated by: min distance from this node to other feasible nodes
    #    - distance from current to this node
    #    If this is large positive, the node is well-connected to others and we should visit it soon
    #    Actually, let's think again...
    #
    # Let's use a simpler but effective heuristic:
    # Score = distance from current + alpha * (average distance from this node to remaining feasible nodes)
    # This encourages visiting nodes that are close to current AND close to other unvisited nodes
    # (reducing future travel costs)
    
    n_feasible = len(feasible_nodes)
    
    # Distance from current to each feasible node
    dist_from_current = dists
    
    # For each feasible node, compute average distance to other feasible nodes
    # This is an n^2 operation but n is at most 50
    if n_feasible > 1:
        # Build sub-matrix of distances between feasible nodes
        sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        # For each node, average distance to all other feasible nodes (excluding self)
        np.fill_diagonal(sub_dist, 0)
        avg_dist_to_others = sub_dist.sum(axis=1) / (n_feasible - 1)
    else:
        avg_dist_to_others = np.array([0.0])
    
    # Normalize distances to make them comparable
    max_dist_current = np.max(dist_from_current) if n_feasible > 0 else 1.0
    max_avg_dist = np.max(avg_dist_to_others) if n_feasible > 1 else 1.0
    
    if max_dist_current == 0:
        max_dist_current = 1.0
    if max_avg_dist == 0:
        max_avg_dist = 1.0
    
    # Score: lower is better
    # Weight the distance from current more heavily, but also consider future connectivity
    alpha = 0.3
    score = dist_from_current / max_dist_current + alpha * (avg_dist_to_others / max_avg_dist)
    
    # Also consider capacity: prefer nodes with smaller demand when capacity is tight
    # This helps avoid getting stuck with large demands later
    demand_vals = demands[feasible_nodes]
    max_demand = np.max(demand_vals) if len(demand_vals) > 0 else 1.0
    if max_demand == 0:
        max_demand = 1.0
    
    # Capacity pressure: if rest_capacity is small relative to remaining demands,
    # prefer smaller demands
    remaining_demand = np.sum(demands[unvisited_arr])
    if remaining_demand > 0:
        capacity_ratio = rest_capacity / remaining_demand
        if capacity_ratio < 1.0:
            # Tight capacity, penalize large demands
            beta = 0.2 * (1.0 - capacity_ratio)
            score += beta * (demand_vals / max_demand)
    
    # Select the node with minimum score
    best_idx = np.argmin(score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
