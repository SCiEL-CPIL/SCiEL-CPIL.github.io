"""Lookahead heuristic for Capacitated VRP."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a 2-step lookahead.
    For each feasible candidate, estimates the cost of going to it and then to 
    its best feasible successor (or back to depot if no successor fits).
    """
    if not unvisited_nodes:
        return depot

    # Convert to numpy array for vectorized operations if possible, 
    # but loop is fine for 50 nodes.
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # 1. Filter feasible nodes based on capacity
    # demands is a dict or list? The signature implies standard access. 
    # Assuming demands is a dict or list where demands[node] works.
    # The prompt says "demands, distance_matrix". Usually demands is a list/dict.
    # Let's assume demands is a dict or list indexable by node id.
    
    feasible_nodes = []
    for node in unvisited:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if not feasible_nodes:
        return depot

    feasible_nodes = np.array(feasible_nodes, dtype=int)
    
    # 2. Calculate immediate distance from current_node to each feasible node
    dists_to_candidates = distance_matrix[current_node, feasible_nodes]
    
    best_node = -1
    best_score = float('inf')
    
    # Pre-compute distances from depot to all nodes? 
    # dist_depot = distance_matrix[depot, :]
    
    for i, cand in enumerate(feasible_nodes):
        d1 = dists_to_candidates[i]
        
        # Lookahead: What is the best next step after visiting cand?
        # Remaining capacity after cand
        rem_cap_after = rest_capacity - demands[cand]
        
        # Find best successor among unvisited nodes (excluding cand) that fits in rem_cap_after
        # We must exclude 'cand' from the potential successors in this lookahead step
        # Actually, standard lookahead considers the set of *other* unvisited nodes.
        
        # Optimization: Instead of filtering the whole list every time, 
        # we can check if there's a cheap successor.
        
        # Get distances from cand to all other unvisited nodes
        # Note: unvisited includes cand. We should exclude cand.
        others_mask = unvisited != cand
        others = unvisited[others_mask]
        
        if len(others) == 0:
            # No other nodes left, must return to depot
            d2 = distance_matrix[cand, depot]
        else:
            # Filter others by capacity
            # Vectorized check
            other_demands = np.array([demands[o] for o in others])
            feasible_others_mask = other_demands <= rem_cap_after
            
            if not np.any(feasible_others_mask):
                # No feasible successor, must return to depot
                d2 = distance_matrix[cand, depot]
            else:
                feasible_others = others[feasible_others_mask]
                # Find the one with minimum distance from cand
                dists_from_cand = distance_matrix[cand, feasible_others]
                d2 = np.min(dists_from_cand)
        
        total_cost = d1 + d2
        
        # Tie-breaker: Prefer smaller demand? Or just min cost.
        if total_cost < best_score:
            best_score = total_cost
            best_node = cand

    if best_node == -1:
        return depot
        
    return int(best_node)
# EVOLVE-BLOCK-END
