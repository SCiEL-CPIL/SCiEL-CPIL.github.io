"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 110  # 120s limit, keep 10s margin

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a 2-step lookahead greedy heuristic.
    For each feasible candidate, estimates the cost of moving to it and then
    to its best possible next move (either another customer or the depot).
    """
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        # Fallback: return depot to close route safely
        return 0

    if not unvisited_nodes:
        return 0

    # Filter feasible nodes based on capacity
    # demands is likely an array where index is node id
    # unvisited_nodes is a list or array of node ids
    feasible_nodes = []
    for u in unvisited_nodes:
        if demands[u] <= rest_capacity:
            feasible_nodes.append(u)
    
    if not feasible_nodes:
        return 0

    # If only one feasible node, return it (unless depot is better? No, must visit if feasible)
    # Actually, if only one is feasible, we must go there or return to depot.
    # The heuristic will handle this.

    best_node = 0
    best_score = float('inf')
    
    dm = distance_matrix
    c = current_node
    
    # Pre-compute distances from current node to all feasible nodes
    # We need to evaluate each candidate u
    
    # To speed up, we can vectorize the "nearest neighbor from u" part if possible,
    # but the capacity constraint on the *next* node depends on u's demand.
    # So we loop through feasible nodes.
    
    # Optimization: If the number of feasible nodes is small, simple loop is fine.
    # With N=50, max 50 iterations.
    
    for u in feasible_nodes:
        d_cu = dm[c, u]
        
        # Calculate the best next step from u
        # The next step can be:
        # 1. The depot (0)
        # 2. Another node v in unvisited_nodes such that demands[v] <= rest_capacity - demands[u]
        
        rem_cap = rest_capacity - demands[u]
        
        # Best distance to depot
        d_u_depot = dm[u, depot]
        
        # Best distance to another feasible node
        # We need min dm[u, v] for v in unvisited_nodes where v != u and demands[v] <= rem_cap
        min_d_u_v = float('inf')
        
        # Iterate through unvisited nodes to find best v
        # This is O(N) per u, so O(N^2) total.
        for v in unvisited_nodes:
            if v == u:
                continue
            if demands[v] <= rem_cap:
                d_uv = dm[u, v]
                if d_uv < min_d_u_v:
                    min_d_u_v = d_uv
        
        # The best next move cost from u
        if min_d_u_v < d_u_depot:
            d_u_next = min_d_u_v
        else:
            d_u_next = d_u_depot
            
        total_score = d_cu + d_u_next
        
        # Tie breaker: prefer lower demand if scores are close?
        # Or prefer closer to depot if capacity is tight?
        # Let's add a small penalty for high demand if capacity is very tight (< 20% remaining)
        if rest_capacity < 10: # Arbitrary threshold for "tight"
             total_score += 0.1 * (demands[u] / rest_capacity)
        
        if total_score < best_score:
            best_score = total_score
            best_node = u
            
    return int(best_node)

# EVOLVE-BLOCK-END
