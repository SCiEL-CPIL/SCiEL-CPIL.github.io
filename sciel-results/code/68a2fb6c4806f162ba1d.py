"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0  # safety margin of 2s before 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing: select next node or return to depot.
    
    Strategy:
    1. Filter to feasible nodes (demand <= rest_capacity).
    2. If none feasible, return depot.
    3. Use a regret-based lookahead score to pick the best next node.
       The score balances immediate distance, future connectivity, and 
       penalty for nodes that are hard to reach later.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if not np.any(feasible_mask):
            return int(depot)
        feasible_nodes = unvisited_nodes[feasible_mask]
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Filter to feasible nodes
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Compute scores for each feasible node
    n = len(feasible_nodes)
    scores = np.zeros(n)
    
    for i, node in enumerate(feasible_nodes):
        # Immediate distance from current node
        d_immediate = distance_matrix[current_node, node]
        
        # Lookahead: estimate cost of going from this node back to depot
        # plus average distance from this node to other unvisited feasible nodes
        d_to_depot = distance_matrix[node, depot]
        
        # Average distance from this node to other feasible unvisited nodes
        if n > 1:
            other_nodes = np.delete(feasible_nodes, i)
            avg_dist_to_others = np.mean(distance_matrix[node, other_nodes])
        else:
            avg_dist_to_others = 0.0
        
        # Regret: how much worse is this node compared to the best alternative?
        # We approximate regret by looking at the second-best immediate distance
        # among feasible nodes, relative to this node's immediate distance.
        
        # Combined score: lower is better
        # Weight immediate distance heavily, add lookahead penalty
        # The idea: prefer nodes that are close now AND have good connectivity
        # for the remaining route.
        
        # Penalty for nodes that are far from depot (hard to close route)
        depot_penalty = 0.3 * d_to_depot
        
        # Penalty for nodes that are isolated from other unvisited nodes
        isolation_penalty = 0.2 * avg_dist_to_others
        
        # Capacity consideration: prefer nodes with lower demand when capacity
        # is tight, to preserve flexibility
        capacity_ratio = rest_capacity / max(1, demands[node])
        # Lower capacity_ratio means tighter fit, so penalize
        if capacity_ratio < 1.5:
            capacity_penalty = 0.1 * (1.5 - capacity_ratio) * d_immediate
        else:
            capacity_penalty = 0.0
        
        scores[i] = d_immediate + depot_penalty + isolation_penalty + capacity_penalty
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
