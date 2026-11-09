import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Clarke-Wright savings-based heuristic with adaptive depot penalty.
    
    Score for each candidate v:
      savings = d(current, v) + d(v, best_next_or_depot) - d(current, best_next_or_depot)
    where best_next is the node that maximizes the savings when chained after v.
    Adaptive penalty: as rest_capacity decreases, we add a penalty to discourage
    visiting nodes far from depot (to avoid long tails).
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited_arr = np.array(unvisited_nodes)
    n = len(unvisited_arr)
    
    # Filter feasible candidates based on capacity
    demands_unvisited = demands[unvisited_arr]
    feasible_mask = demands_unvisited <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_arr[feasible_mask]
    m = len(feasible_nodes)
    
    # Precompute distances from current to all feasible nodes
    d_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Adaptive depot penalty: tightens as capacity empties
    # When rest_capacity is low, we penalize nodes far from depot more heavily
    cap_ratio = rest_capacity / 50.0  # capacity is 50
    depot_penalty_weight = 1.0 / (1.0 + cap_ratio)  # 0.5 when full, 1.0 when empty
    
    best_score = -np.inf
    best_node = feasible_nodes[0]
    
    for i, v in enumerate(feasible_nodes):
        d_cv = d_current_to_feasible[i]
        
        # Find the best next node u (or depot) to chain after v
        # u must be feasible from v: demands[u] <= rest_capacity - demands[v]
        new_rest = rest_capacity - demands[v]
        
        min_chain_cost = np.inf
        best_next = depot
        
        # Check all feasible nodes u != v
        for j, u in enumerate(feasible_nodes):
            if u == v:
                continue
            if demands[u] > new_rest:
                continue
            d_vu = distance_matrix[v, u]
            # The chain cost is d(v, u), but we want to minimize the total route length
            # Savings perspective: visiting v then u saves d(current, v) + d(v, u) - d(current, u)
            # But for scoring, we want to estimate the marginal cost of adding v to the route
            # which is d(current, v) + d(v, best_next) - d(current, best_next)
            # We approximate best_next as the nearest feasible node from v, or depot
            if d_vu < min_chain_cost:
                min_chain_cost = d_vu
                best_next = u
        
        # If no feasible u, must go to depot
        if min_chain_cost == np.inf:
            min_chain_cost = distance_matrix[v, depot]
            best_next = depot
        
        # Compute savings-based score
        # Savings = d(current, best_next) - d(current, v) - d(v, best_next)
        # We want to maximize savings, so minimize -savings
        if best_next == depot:
            d_current_best = distance_matrix[current_node, depot]
        else:
            d_current_best = distance_matrix[current_node, best_next]
        
        savings = d_current_best - d_cv - min_chain_cost
        
        # Add adaptive depot penalty: penalize nodes far from depot when capacity is low
        d_v_depot = distance_matrix[v, depot]
        penalty = depot_penalty_weight * d_v_depot
        
        # Score: maximize savings, minimize penalty
        # Equivalent to minimizing: -savings + penalty
        score = -savings + penalty
        
        if score < best_score:
            best_score = score
            best_node = v
    
    return int(best_node)
