"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    # If only 1 node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # If few nodes left, use exact brute force to optimize the tail
    # This ensures the final connections (including return to destination) are optimal
    if n_unvisited <= 5:
        nodes = list(unvisited_nodes)
        best_cost = np.inf
        best_first_node = nodes[0]
        
        for perm in permutations(nodes):
            # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]
                
        return int(best_first_node)

    # Heuristic for larger sets: 2-step lookahead
    # 1. Distance from current to all unvisited
    dist_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance between unvisited nodes
    # Calculate min distance from each unvisited node to any OTHER unvisited node
    D_ii = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Exclude self by setting diagonal to inf
    np.fill_diagonal(D_ii, np.inf)
    
    # Min distance to any other unvisited node
    min_future = np.min(D_ii, axis=1)
    
    # Score: Immediate distance + Future connectivity cost
    scores = dist_curr + min_future
    
    # Select the node with the minimum score
    idx = np.argmin(scores)
    
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
