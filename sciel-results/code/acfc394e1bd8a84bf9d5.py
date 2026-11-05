"""Improved constructive TSP heuristic with 3-step lookahead for final steps."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Picks the next city in a TSP tour using a lookahead heuristic.
    
    For the last few cities, it evaluates exact permutations to minimize
    the cost including the return to the destination. For earlier steps,
    it uses a 2-step lookahead to avoid isolated nodes.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
        
    if n == 1:
        return int(unv[0])
        
    if n == 2:
        u1, u2 = int(unv[0]), int(unv[1])
        # Path 1: c -> u1 -> u2 -> dest
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        # Path 2: c -> u2 -> u1 -> dest
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        return u1 if cost1 <= cost2 else u2

    if n == 3:
        u1, u2, u3 = int(unv[0]), int(unv[1]), int(unv[2])
        candidates = [u1, u2, u3]
        best_cost = np.inf
        best_node = u1
        
        for u in candidates:
            others = [v for v in candidates if v != u]
            v1, v2 = others[0], others[1]
            
            # Option A: c -> u -> v1 -> v2 -> dest
            c1 = distance_matrix[current_node, u] + distance_matrix[u, v1] + distance_matrix[v1, v2] + distance_matrix[v2, destination_node]
            # Option B: c -> u -> v2 -> v1 -> dest
            c2 = distance_matrix[current_node, u] + distance_matrix[u, v2] + distance_matrix[v2, v1] + distance_matrix[v1, destination_node]
            
            min_cost_for_u = min(c1, c2)
            
            if min_cost_for_u < best_cost:
                best_cost = min_cost_for_u
                best_node = u
                
        return best_node

    # n > 3: 2-step lookahead
    d_cur = distance_matrix[current_node, unv]
    
    # Submatrix of distances among unvisited nodes
    sub_dm = distance_matrix[np.ix_(unv, unv)]
    np.fill_diagonal(sub_dm, np.inf)
    
    # Min distance from each unvisited node to any other unvisited node
    min_onward = np.min(sub_dm, axis=1)
    
    # Score: immediate cost + estimated future cost
    scores = d_cur + min_onward
    
    best_idx = np.argmin(scores)
    
    return int(unv[best_idx])
# EVOLVE-BLOCK-END
