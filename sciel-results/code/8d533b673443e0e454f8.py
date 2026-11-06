"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Lookahead Cost Minimization.
    
    Selects the next customer by minimizing the immediate travel cost plus the 
    estimated cost to return to the depot from that customer. This approximates 
    the total length added to the current route by visiting this node, favoring 
    nodes that allow for a shorter total route segment.
    
    This is effectively a greedy approach for minimizing the total route length 
    of the current trip, assuming we will return to the depot after this node 
    (or that the return cost is a dominant factor in the decision).
    
    Score(u) = distance(current, u) + distance(u, depot)
    We want to minimize this score among feasible nodes.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if running low on time, fall back to nearest neighbor for speed
    if _time_exceeded():
        # Nearest neighbor fallback
        dm = distance_matrix
        c = current_node
        u_arr = np.array(feasible)
        dist_c_u = dm[c, u_arr]
        min_idx = np.argmin(dist_c_u)
        return int(u_arr[min_idx])

    dm = distance_matrix
    c = current_node
    d = depot
    
    u_arr = np.array(feasible)
    
    # Calculate total cost if we go to u and then return to depot
    # This is a proxy for the "extension cost" of the route
    dist_c_u = dm[c, u_arr]
    dist_u_d = dm[u_arr, d]
    
    # The objective is to minimize the total length of the route.
    # By choosing u that minimizes d(c,u) + d(u,d), we are trying to close the loop efficiently.
    # This is different from pure Savings (which is d(c,u) - d(d,u)) or Nearest Neighbor (d(c,u)).
    # It balances proximity to current node with proximity to depot.
    
    scores = dist_c_u + dist_u_d
    
    min_idx = np.argmin(scores)
    
    return int(u_arr[min_idx])

# EVOLVE-BLOCK-END
