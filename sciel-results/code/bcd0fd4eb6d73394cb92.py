"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _DEADLINE_START

def _time_budget_remaining():
    return max(0.0, _DEADLINE_LIMIT - _elapsed())

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If no unvisited nodes, return depot
    if len(unvisited_nodes) == 0:
        return depot

    # If time is critically low, just return nearest feasible
    if _time_budget_remaining() < 0.05:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    dm = np.asarray(distance_matrix)
    dem = np.asarray(demands)
    
    # Get feasible nodes (capacity constraint)
    feasible_mask = dem[unvisited_arr] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node to feasible nodes
    dists = dm[current_node, feasible_nodes]
    
    # Also compute distance from each feasible node back to depot
    dist_to_depot = dm[feasible_nodes, depot]
    
    # Score: we want to minimize total route length. 
    # A good heuristic: minimize (dist_to_node + dist_node_to_depot) / (1 + demand)
    # This balances proximity, return cost, and capacity usage.
    # Higher demand means less capacity left for future nodes, so prefer lower demand if distances are similar.
    
    # Try a composite score: distance from current + distance to depot, weighted by remaining capacity consideration
    # We want to fill the truck efficiently, so prefer nodes that are close and have moderate demand
    
    # Score = dists + dist_to_depot * alpha - beta * demand (we want to minimize this)
    # But we also need to consider that picking a high-demand node early might be good if it's close
    
    # Let's use: score = dists + dist_to_depot * 0.5 + dem[feasible_nodes] * 0.1
    # This penalizes far nodes, nodes far from depot, and high-demand nodes slightly
    
    dem_f = dem[feasible_nodes]
    scores = dists + dist_to_depot * 0.3 + dem_f * 0.2
    
    # But we also want to consider the "regret" - how much worse would it be to not pick this node
    # For now, use a simple composite score
    
    # Actually, let's think about this differently. The key insight for CVRP is:
    # 1. Don't waste capacity - pack trucks well
    # 2. Minimize travel distance
    # 3. Avoid long detours
    
    # A better heuristic: minimize (dist_to_node + dist_node_to_depot) but with a 
    # penalty for leaving too much capacity unused. If we have lots of capacity left,
    # prefer higher demand nodes. If little capacity left, prefer lower demand.
    
    # Capacity utilization factor: if rest_capacity is large relative to max demand,
    # we can afford to pick high-demand nodes. If small, pick low-demand.
    
    max_demand = np.max(dem) if len(dem) > 0 else 1
    cap_ratio = rest_capacity / max(max_demand, 1)
    
    # When capacity is plentiful (high cap_ratio), we can be more aggressive with demand
    # When capacity is tight, prefer low demand
    
    # Adjust the demand penalty based on capacity situation
    # If cap_ratio > 2, we have lots of room, so demand penalty should be lower
    # If cap_ratio < 1.5, we're tight, so demand penalty should be higher
    
    demand_penalty_weight = 0.5 / max(cap_ratio, 1.0)
    
    scores = dists + dist_to_depot * 0.3 + dem_f * demand_penalty_weight
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
