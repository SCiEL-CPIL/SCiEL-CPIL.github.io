"""Direct heuristic for TSP next-node selection."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Combined structural approach:
    1. Early phase (n > 35): Pure Nearest Neighbor for fast, locally optimal tour growth.
    2. Mid phase (10 < n <= 35): Blend current distance, destination distance, and
       a connectivity penalty to avoid isolating far-away clusters.
    3. Late phase (n <= 10): Full nearest-neighbor lookahead to minimize the remaining
       sub-tour cost, ensuring a short closing edge.
    
    This combines the phase-aware structure of the current program with the
    lookahead refinement from the inspirations.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        u, v = unvisited_nodes
        # Cost of closing via u then v, or v then u
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Phase 1: Pure Nearest Neighbor for large n
    if n_unvisited > 35:
        d_curr = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_curr)
        return int(unvisited_nodes[best_idx])
    
    # Phase 3: Full lookahead for small n (n <= 10)
    if n_unvisited <= 10:
        lookahead_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            # Build a small tour: start at candidate i, NN through remaining unvisited, then to dest
            remaining = list(unvisited_nodes)
            remaining.pop(i)  # remove candidate i
            cur = unvisited_nodes[i]
            tour_cost = distance_matrix[current_node, unvisited_nodes[i]]  # edge to candidate
            # NN from candidate through remaining
            while remaining:
                # find nearest unvisited from cur
                d_to_rem = np.array([distance_matrix[cur, r] for r in remaining])
                j = int(np.argmin(d_to_rem))
                nxt = remaining[j]
                tour_cost += d_to_rem[j]
                remaining.pop(j)
                cur = nxt
            # edge from last to destination
            tour_cost += distance_matrix[cur, destination_node]
            lookahead_costs[i] = tour_cost
        
        best_idx = np.argmin(lookahead_costs)
        return int(unvisited_nodes[best_idx])
    
    # Phase 2: Mid-phase blend with connectivity penalty (10 < n <= 35)
    # Adaptive weight: more emphasis on destination as we get closer to finishing
    # At n=35, alpha=0.1; at n=10, alpha=0.6
    alpha = 0.1 + 0.5 * (35.0 - n_unvisited) / 25.0
    
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Connectivity penalty: penalize nodes that are far from other unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(d_sub, np.inf)
    d_connect = np.min(d_sub, axis=1)
    
    min_c = np.min(d_connect)
    max_c = np.max(d_connect)
    if max_c > min_c:
        connect_norm = (d_connect - min_c) / (max_c - min_c)
    else:
        connect_norm = np.zeros(n_unvisited)
    
    beta = 0.2  # weight on connectivity penalty
    scores = d_curr + alpha * d_dest + beta * connect_norm * (max_c - min_c)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
