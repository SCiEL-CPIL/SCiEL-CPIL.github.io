"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_DEADLINE = None
_DEADLINE_LIMIT = 115.0  # Safety margin before 120s limit


def _init_deadline():
    global _DEADLINE
    _DEADLINE = time.time() + _DEADLINE_LIMIT


def _time_exceeded():
    global _DEADLINE
    if _DEADLINE is None:
        _init_deadline()
    return time.time() >= _DEADLINE


def _safety_return(unvisited_nodes, rest_capacity, demands, current_node, depot, distance_matrix):
    """Fallback: return nearest feasible node or depot."""
    if len(unvisited_nodes) == 0:
        return depot
    unvisited = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    feasible_nodes = unvisited[feasible_mask]
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    dists = distance_matrix[current_node, feasible_nodes]
    return int(feasible_nodes[np.argmin(dists)])


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Initialize deadline on first call
    global _DEADLINE
    if _DEADLINE is None:
        _init_deadline()
    
    # Check time
    if _time_exceeded():
        return _safety_return(unvisited_nodes, rest_capacity, demands, current_node, depot, distance_matrix)
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # If time is getting tight, use simple heuristic
    if _time_exceeded():
        return _safety_return(unvisited_nodes, rest_capacity, demands, current_node, depot, distance_matrix)
    
    # Compute distances from current node to feasible nodes
    dist_to_current = distance_matrix[current_node, feasible_nodes]
    
    # Compute distance from depot for each feasible node (for return cost estimation)
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Lookahead: for each candidate i, estimate the cost of going to the next nearest feasible node after i
    # Use a 1-step lookahead with soft minimum
    n = len(feasible_nodes)
    scores = np.zeros(n)
    
    for i in range(n):
        # Cost to go from current to candidate i
        cost_i = dist_to_current[i]
        
        # After visiting i, remaining capacity
        rest_cap_after = rest_capacity - demands[feasible_nodes[i]]
        
        # Find feasible nodes after visiting i (excluding i itself)
        if rest_cap_after <= 0:
            # Must return to depot
            lookahead_cost = dist_to_depot[i]
        else:
            # Find other feasible nodes
            other_mask = np.ones(n, dtype=bool)
            other_mask[i] = False
            other_feasible = feasible_nodes[other_mask]
            
            if len(other_feasible) == 0:
                lookahead_cost = dist_to_depot[i]
            else:
                # Check which of these are feasible with reduced capacity
                feasible_after_mask = demands[other_feasible] <= rest_cap_after
                if not np.any(feasible_after_mask):
                    lookahead_cost = dist_to_depot[i]
                else:
                    feasible_after = other_feasible[feasible_after_mask]
                    # Distance from candidate i to nearest feasible-after node
                    dists_from_i = distance_matrix[feasible_nodes[i], feasible_after]
                    min_next_dist = np.min(dists_from_i)
                    # Also consider returning to depot
                    lookahead_cost = min(min_next_dist, dist_to_depot[i])
        
        scores[i] = cost_i + lookahead_cost
    
    # Add a small penalty for high demand nodes to prefer lighter loads when costs are close
    # This helps balance routes better
    demand_penalty = 0.01 * demands[feasible_nodes]
    scores += demand_penalty
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
