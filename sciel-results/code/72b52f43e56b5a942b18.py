"""Savings-based heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings-based heuristic.
    Returns 0 (depot) if no feasible node can be added or if time limit is reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # 2. Calculate savings score
    # Savings = dist(current, depot) + dist(depot, candidate) - dist(current, candidate)
    # High savings means it's better to visit the candidate now than to return to depot and start a new route.
    # Note: In a standard greedy savings approach, we usually look at the total cost.
    # However, since we are building a route sequentially, we want to minimize the incremental cost.
    # Incremental cost to visit candidate c now: dist(current, c)
    # Incremental cost if we close route and start new one: dist(current, depot) + dist(depot, c)
    # So we want to minimize: dist(current, c) - (dist(current, depot) + dist(depot, c))
    # Which is equivalent to maximizing: dist(current, depot) + dist(depot, c) - dist(current, c)
    
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_cand = distance_matrix[depot, feasible]
    d_cur_cand = distance_matrix[current_node, feasible]
    
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # 3. Heuristic adjustment: 
    # Pure savings can sometimes lead to "zig-zag" patterns if not careful, but it generally clusters well.
    # To avoid getting stuck in local optima or long detours, we can blend with nearest neighbor slightly.
    # However, the prompt notes that savings-based scoring with alpha=0.9 (prioritizing far nodes) improved scores.
    # This suggests that maximizing savings (which favors nodes that reduce the "penalty" of returning to depot) is effective.
    
    # Let's stick to pure savings maximization, as it is a distinct strategy from pure nearest-neighbor.
    # If savings are very low, it might be better to return to depot to start a new route?
    # Actually, the problem asks to pick the NEXT customer. If no customer yields positive savings 
    # (i.e., visiting it is more expensive than closing the route and starting a new one), 
    # should we return to depot?
    # The interface allows returning 0.
    
    # Let's check if the best savings is positive.
    best_savings_idx = np.argmax(savings)
    best_savings = savings[best_savings_idx]
    
    # If the best savings is very low or negative, it implies that closing the route is better.
    # However, in CVRP, we usually must visit all nodes. Returning to depot is only valid if we have finished 
    # or if we are forced to. But here we have unvisited nodes. 
    # If we return to depot, we start a new route. 
    # The cost comparison is:
    # Option A: Go to node c. Cost added: d(cur, c). Remaining cap: cap - d(c).
    # Option B: Go to depot. Cost added: d(cur, dep). Then start new route from depot to some node.
    # The "savings" calculation above effectively compares the marginal cost.
    # If savings > 0, it is cheaper to go to c than to go to depot (in terms of the specific leg).
    # But we also have to consider future costs. A simple greedy savings might be myopic.
    
    # Given the previous finding that "Savings-based scoring... prioritizes nodes far from depot... reduced vehicle count",
    # let's select the node with maximum savings.
    
    return int(feasible[best_savings_idx])

# EVOLVE-BLOCK-END
