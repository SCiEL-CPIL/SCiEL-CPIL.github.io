"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (150s wall-clock, use ~145s safety margin)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0


def _elapsed_ok():
    return time.time() < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for CVRP: pick next customer or return to depot.
    
    Uses a composite score that balances:
    1. Distance to next node (proximity)
    2. Distance from next node back to depot (route closure cost)
    3. Demand feasibility
    4. A slight bias toward closing routes when remaining capacity is low
    """
    if not _elapsed_ok():
        return 0  # Return to depot on timeout
    
    if len(unvisited_nodes) == 0:
        return 0
    
    # Check if we can visit any more nodes given capacity
    # demands is indexed by node id
    feasible = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if len(feasible) == 0:
        return 0
    
    # If only one feasible node, go there (or depot if it's better)
    if len(feasible) == 1:
        node = feasible[0]
        # Compare: go to node then depot vs just depot
        cost_go_node = distance_matrix[current_node, node] + distance_matrix[node, depot]
        cost_go_depot = distance_matrix[current_node, depot]
        if cost_go_depot <= cost_go_node * 0.95:  # If depot is significantly better
            return 0
        return int(node)
    
    # Multi-criteria scoring for each feasible node
    # For each candidate, estimate the incremental route cost:
    #   d(current, node) + d(node, depot)  [if this is the last node]
    # vs.
    #   d(current, node)  [if more nodes follow]
    
    # We'll use a blend:
    # score = alpha * d(current, node) + beta * d(node, depot) + gamma * penalty
    
    # Adaptive parameters based on remaining capacity and number of unvisited
    n_unvisited = len(unvisited_nodes)
    total_remaining_demand = sum(demands[n] for n in feasible)
    
    # If remaining capacity is tight relative to total demand, favor closing
    capacity_ratio = rest_capacity / (total_remaining_demand + 1e-9)
    
    # Weight for depot-return cost: higher when capacity is tight
    beta = 0.3 + 0.4 * (1.0 - min(capacity_ratio, 1.0))
    alpha = 1.0 - beta * 0.5  # Keep alpha dominant
    
    # Small penalty for large demands when capacity is tight (prefer smaller loads later)
    gamma = 0.1 * (1.0 - min(capacity_ratio, 1.0))
    
    scores = []
    for node in feasible:
        d_cur_node = distance_matrix[current_node, node]
        d_node_depot = distance_matrix[node, depot]
        demand_penalty = demands[node] * gamma
        
        # Estimated total route segment cost
        score = alpha * d_cur_node + beta * d_node_depot + demand_penalty
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmin(scores)
    best_node = feasible[best_idx]
    
    # Additional check: if going to depot now is clearly better than any option
    cost_depot = distance_matrix[current_node, depot]
    best_score = scores[best_idx]
    
    # If depot is cheaper than the best candidate's incremental cost, close route
    # Use a threshold to avoid premature closure
    if cost_depot < best_score * 0.85 and n_unvisited <= 5:
        return 0
    
    return int(best_node)
# EVOLVE-BLOCK-END
