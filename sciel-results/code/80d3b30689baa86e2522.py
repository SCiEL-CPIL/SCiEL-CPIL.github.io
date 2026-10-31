"""Rewritten program: TSP constructive heuristic using 2-step regret-based lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TOTAL_LIMIT = 150.0
_MARGIN = 5.0

def _is_deadline_passed():
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: 2-step regret-based lookahead.
    
    For each candidate u in unvisited_nodes, compute:
      - immediate cost: d(current, u)
      - 2-step lookahead: min over v in unvisited\{u} of d(u, v) + d(v, destination)
        (or d(u, v) if destination is far)
      - regret: how much worse the 2nd-best continuation is vs the best
        (high regret = this node is "forced", waiting would be costly)
    
    Score = alpha * d(current,u) + beta * lookahead_cost + gamma * regret
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        u0 = int(unvisited_nodes[0])
        u1 = int(unvisited_nodes[1])
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        return u0 if cost0 <= cost1 else u1
        
    if n == 3:
        # Enumerate all permutations of 3 nodes
        u0, u1, u2 = int(unvisited_nodes[0]), int(unvisited_nodes[1]), int(unvisited_nodes[2])
        # Try all 6 orderings, pick the one that starts with the best first choice
        # Actually, we just need to pick the best first node given optimal completion
        best_cost = np.inf
        best_first = u0
        for first in [u0, u1, u2]:
            remaining = [x for x in [u0, u1, u2] if x != first]
            r0, r1 = remaining[0], remaining[1]
            cost_a = distance_matrix[current_node, first] + distance_matrix[first, r0] + distance_matrix[r0, r1] + distance_matrix[r1, destination_node]
            cost_b = distance_matrix[current_node, first] + distance_matrix[first, r1] + distance_matrix[r1, r0] + distance_matrix[r0, destination_node]
            total = min(cost_a, cost_b)
            if total < best_cost:
                best_cost = total
                best_first = first
        return best_first
        
    # Deadline guard: fall back to nearest neighbor
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # Precompute distances
    d_curr = distance_matrix[current_node, unvisited_nodes]  # shape (n,)
    d_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n,)
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # shape (n, n)
    
    # For each candidate i, compute:
    #   lookahead_i = min over j != i of d_sub[i,j] + d_dest[j]
    #   regret_i = 2nd_min - min over j != i of d_sub[i,j] + d_dest[j]
    # 
    # This is O(n^2) total which is fine for n up to ~50
    
    lookahead_costs = np.zeros(n)
    regrets = np.zeros(n)
    
    for i in range(n):
        # Costs of going from i to each j (j != i), then to destination
        # d_sub[i, j] + d_dest[j] for all j != i
        cont_costs = d_sub[i].copy()
        cont_costs[i] = np.inf
        cont_costs += d_dest  # now cont_costs[j] = d_sub[i,j] + d_dest[j]
        
        # Find min and 2nd min
        min_idx = np.argmin(cont_costs)
        min_val = cont_costs[min_idx]
        cont_costs[min_idx] = np.inf
        second_min_val = np.min(cont_costs)
        
        lookahead_costs[i] = min_val
        regrets[i] = second_min_val - min_val
    
    # Adaptive weights based on remaining fraction
    remaining_fraction = n / 50.0
    
    # When many nodes remain: weight immediate distance and regret more
    # When few nodes remain: weight lookahead (destination proximity) more
    alpha = 0.4  # weight for immediate distance
    beta = 0.3 + 0.3 * (1.0 - remaining_fraction)  # weight for lookahead, increases as we approach end
    gamma = 0.3 * remaining_fraction  # weight for regret, decreases as we approach end
    
    # Normalize to avoid scale issues
    # Scale each component by its mean
    mean_dcurr = np.mean(d_curr) if n > 0 else 1.0
    mean_look = np.mean(lookahead_costs) if n > 0 else 1.0
    mean_reg = np.mean(regrets) if n > 0 else 1.0
    
    if mean_dcurr < 1e-10:
        mean_dcurr = 1.0
    if mean_look < 1e-10:
        mean_look = 1.0
    if mean_reg < 1e-10:
        mean_reg = 1.0
    
    scores = alpha * (d_curr / mean_dcurr) + beta * (lookahead_costs / mean_look) + gamma * (regrets / mean_reg)
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
