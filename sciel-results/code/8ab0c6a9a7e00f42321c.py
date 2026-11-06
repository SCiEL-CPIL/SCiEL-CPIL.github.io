"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the entire evaluation run to enforce deadline
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    if time.time() - _start_time > 145:
        # Fallback to simple nearest feasible to avoid timeout
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Identify feasible nodes
    unvisited_array = np.array(unvisited_nodes)
    if len(unvisited_array) == 0:
        return depot
        
    feasible_mask = demands[unvisited_array] <= rest_capacity
    feasible_nodes = unvisited_array[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. Regret-based selection (structural change from savings+density)
    # For each feasible node, compute the cost of inserting it at the END of the current route
    # vs. the cost if we were to visit it later (approximated by distance from depot).
    # Regret = second_best_cost - best_cost for each node across "positions"
    # Here we use a simplified regret: for each node, how much worse is it to NOT visit now?
    
    n_feasible = len(feasible_nodes)
    
    # Cost of visiting node j now: distance from current to j
    cost_now = distance_matrix[current_node, feasible_nodes]
    
    # Cost of visiting node j later (after going back to depot or from another node):
    # Approximate by distance from depot to j (if we close and restart) 
    # or by the minimum distance from any other feasible node to j
    cost_later_depot = distance_matrix[depot, feasible_nodes]
    
    # For each node, the "regret" of not picking it now is how much more expensive it becomes
    # We model this as: regret(j) = cost_later(j) - cost_now(j)
    # But we want to pick nodes with HIGH regret (they get much worse later)
    # So we score by: score(j) = -regret(j) + gamma * cost_now(j)
    # Minimizing score = maximizing regret while keeping distance reasonable
    
    # Alternative structural approach: use pairwise regret
    # For each pair (i,j) of feasible nodes, compute the savings of visiting i before j
    # vs j before i. The node that appears as "should go first" in more pairs gets higher priority.
    
    # Simpler regret: for each node j, compute how much it "hurts" if we skip it
    # by looking at the best and second-best insertion positions
    
    # Let's use a different structural model: 
    # Score each node by its "urgency" = how constrained it is by capacity
    # and its "connectivity" = how well it links to remaining nodes
    
    # Urgency: nodes with demand close to rest_capacity are more urgent
    # (they can't be fit in a later route if capacity is tight)
    urgency = demands[feasible_nodes] / max(rest_capacity, 1e-9)
    
    # Connectivity: for each node, sum of distances to all other feasible nodes
    # Lower total distance = better connected = should visit sooner
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    np.fill_diagonal(sub_dist, 0)
    connectivity_cost = np.sum(sub_dist, axis=1)
    
    # Normalize
    if n_feasible > 1:
        conn_min = np.min(connectivity_cost)
        conn_max = np.max(connectivity_cost)
        conn_range = conn_max - conn_min
        if conn_range > 1e-9:
            connectivity_norm = (connectivity_cost - conn_min) / conn_range
        else:
            connectivity_norm = np.zeros(n_feasible)
    else:
        connectivity_norm = np.zeros(n_feasible)
    
    # Distance from current node
    dists = cost_now
    dist_min = np.min(dists)
    dist_max = np.max(dists)
    dist_range = dist_max - dist_min
    if dist_range > 1e-9:
        dist_norm = (dists - dist_min) / dist_range
    else:
        dist_norm = np.zeros(n_feasible)
    
    # Combine: prefer low distance, high urgency, low connectivity cost (well-connected)
    # Score = w1 * dist_norm + w2 * (1 - urgency) + w3 * connectivity_norm
    # Minimize this score
    w1 = 0.4
    w2 = 0.3
    w3 = 0.3
    
    scores = w1 * dist_norm + w2 * (1.0 - urgency) + w3 * connectivity_norm
    
    # Select best node
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
