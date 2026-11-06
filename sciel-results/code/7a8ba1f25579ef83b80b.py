"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def _savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Implements a Savings-based heuristic to select the next node.
    
    1. Identify all "open ends" in the system:
       - The current partial route ends at `current_node` (with remaining capacity `rest_capacity`).
       - Each unvisited node `u` is a potential singleton route starting/ending at `u` (with capacity `demands[u]`).
       - Note: In CVRP, a node is not a route by itself until visited, but for savings calculation,
         we treat the cost of connecting two potential routes as the savings gained.
    
    2. Calculate savings for merging the current route with each unvisited node,
       and for merging each pair of unvisited nodes.
       
       Savings(i, j) = dist(i, depot) + dist(j, depot) - dist(i, j)
       
       However, since we are extending a *current* route, we don't return to depot for the current route yet.
       The standard savings heuristic assumes we start with N+1 routes (depot + each customer) and merge them.
       
       Let's adapt the savings logic for the specific state:
       We have a set of "active" entities:
       - Entity 0: The current route ending at `current_node`. Capacity left: `rest_capacity`.
       - Entity i (for each u in unvisited): A singleton route at `u`. Capacity left: `demands[u]` (actually, it consumes demands[u]).
       
       Wait, the standard savings heuristic constructs the entire solution from scratch.
       Here, we are in the middle of construction. 
       
       Alternative interpretation: 
       The "Savings" heuristic suggests that if we are at node A, we should go to node B if the saving 
       d(A, depot) + d(B, depot) - d(A, B) is high. But this assumes A and B were separate routes.
       
       In our context, `current_node` is part of a route that started at depot. 
       If we go to `u`, we are effectively merging the "current route" with the "singleton u".
       The cost of NOT merging is: finish current route (go to depot) + start new route to u (depot to u).
       Cost of merging: current -> u.
       Saving = (dist(current, depot) + dist(depot, u)) - dist(current, u).
       
       We should pick the `u` that maximizes this saving, subject to capacity constraints.
       
       Additionally, we should consider if merging two *unvisited* nodes would be better?
       No, we can only pick one next node. But the choice of next node affects future savings.
       A pure greedy savings approach picks the `u` that gives the maximum immediate saving.
       
       Let's verify this logic.
       If we pick u, we pay dist(current, u).
       If we didn't pick u, we would eventually pay dist(current, depot) (to close current route) and dist(depot, u) (to visit u later).
       So the immediate benefit of picking u now is:
       Benefit(u) = dist(current, depot) + dist(depot, u) - dist(current, u)
       
       We want to maximize Benefit(u) among feasible u.
       
       This is mathematically equivalent to minimizing:
       dist(current, u) - dist(depot, u)
       (since dist(current, depot) is constant for all u).
       
       Let's test this against Nearest Neighbor.
       NN minimizes dist(current, u).
       Savings minimizes dist(current, u) - dist(depot, u).
       
       If a node u is close to current but far from depot, Savings might prefer another node v that is slightly further from current but much closer to depot?
       Actually, if v is closer to depot, dist(depot, v) is small.
       Let's say:
       u: dist(cur, u)=1, dist(depot, u)=10. Score = 1 - 10 = -9.
       v: dist(cur, v)=2, dist(depot, v)=1. Score = 2 - 1 = 1.
       Savings picks v.
       Cost of picking u: 1. Future cost to return from u to depot: 10. Total: 11.
       Cost of picking v: 2. Future cost to return from v to depot: 1. Total: 3.
       Clearly v is better if we assume we return to depot after the route.
       
       However, we don't return to depot after v necessarily. We might pick more nodes.
       But the savings heuristic is a constructive heuristic that works well because it captures the "efficiency" of the trip.
       
       Let's implement this "Savings Score" approach.
       
       Score(u) = distance_matrix[current_node, u] - distance_matrix[depot, u]
       We want to minimize this score.
       
       We also need to respect capacity.
       
       Is this materially different?
       NN: min d(c, u)
       Savings: min d(c, u) - d(0, u)
       
       This is a different objective function. It biases towards nodes that are "close to the depot" relative to the current position, effectively encouraging routes that don't stray too far from the depot or that complete the loop efficiently.
       
       Let's add a tie-breaker or secondary criterion if needed, but let's stick to the core savings idea first.
       
       To make it more robust, we can combine it with a small penalty for large demands to encourage filling the truck?
       The prompt mentions "capacity-aware tie-breaking fills trucks more completely".
       Savings doesn't inherently fill trucks, but efficient routes often do.
       
       Let's refine the score:
       Primary: Savings (minimize d(c,u) - d(0,u))
       Secondary: Maximize demand (to fill capacity)
       
       Or simply:
       Score = d(c,u) - d(0,u) + lambda * (max_demand - demand[u]) ?
       
       Let's stick to the pure savings first, as it's a distinct, well-known heuristic family.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    dm = distance_matrix
    c = current_node
    d = depot
    
    # Calculate savings score for each feasible node
    # Score = dm[c, u] - dm[d, u]
    # We want to minimize this.
    
    u_arr = np.array(feasible)
    dist_c_u = dm[c, u_arr]
    dist_d_u = dm[d, u_arr]
    
    scores = dist_c_u - dist_d_u
    
    # To incorporate "filling the truck", we can add a small penalty for low demand
    # or a bonus for high demand.
    # Let's try a weighted sum:
    # Final Score = scores - alpha * demands[u_arr]
    # This prefers nodes with higher demand (if alpha is positive enough) or
    # we can just use the savings score.
    
    # Let's try pure savings first.
    min_idx = np.argmin(scores)
    
    return int(u_arr[min_idx])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Savings-based Node Selection.
    
    Selects the next customer that maximizes the "savings" of merging the current
    partial route with that customer's singleton route.
    Savings = dist(current, depot) + dist(depot, customer) - dist(current, customer).
    Maximizing savings is equivalent to minimizing dist(current, customer) - dist(depot, customer).
    """
    return _savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
