"""Savings-based heuristic with demand and distance adjustments for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

# Global cache for precomputed constants
_max_demand = None
_max_dist = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a modified savings heuristic.
    Incorporates demand urgency and distance penalties.
    """
    global _max_demand, _max_dist
    
    # Initialize constants if not done
    if _max_demand is None:
        _max_demand = np.max(demands)
        _max_dist = np.max(distance_matrix)
        if _max_demand == 0: _max_demand = 1
        if _max_dist == 0: _max_dist = 1

    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # 1. Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Pre-compute demand ratio and distance ratios
    d_j = demands[feasible_nodes]
    demand_ratio = d_j / _max_demand
    
    if current_node == depot:
        # At depot: Minimize distance, but boost high demand
        d_c_j = distance_matrix[current_node, feasible_nodes]
        dist_ratio = d_c_j / _max_dist
        
        # Score: -distance + lambda * demand
        # We want to maximize score, so negative distance is good.
        # Let's use a weight for demand to compete with distance.
        # If distance is ~10 and demand is ~1, we need lambda to be significant.
        # Let's try lambda = 1.0 * max_dist? No, let's keep it simple.
        # Score = -d_c_j + 0.5 * demand_ratio
        score = -d_c_j + 0.5 * demand_ratio
        
        max_score_idx = np.argmax(score)
        return int(feasible_nodes[max_score_idx])

    # 2. Calculate savings for each feasible node
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    savings = d_current_depot + d_depot_j - d_current_j
    
    # Add demand bonus and distance penalty
    dist_ratio = d_current_j / _max_dist
    
    # Tuning parameters:
    # lambda_d: weight for demand (0.2 seems safe)
    # mu_d: weight for distance penalty (0.1 seems safe)
    score = savings + 0.2 * demand_ratio - 0.1 * dist_ratio
    
    # 3. If the best score is very low (worse than closing route), return depot
    # Original check: max(savings) < 0
    # With adjustments, the threshold might shift.
    # If score is negative, it implies the "cost" of going to j is higher than going to depot and then j?
    # Actually, savings < 0 means d_c_j > d_c_0 + d_0_j.
    # If score < 0, it's even worse.
    # Let's stick to checking if the best score is < 0.
    if np.max(score) < 0:
        return depot
    
    max_score_idx = np.argmax(score)
    
    return int(feasible_nodes[max_score_idx])
# EVOLVE-BLOCK-END
