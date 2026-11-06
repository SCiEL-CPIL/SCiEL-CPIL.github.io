"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Nearest Neighbor heuristic with 2-step lookahead.
    
    For each unvisited node, it estimates the cost of moving to it plus the cost
    of moving from that node to its nearest neighbor in the remaining unvisited set
    (or to the destination if it's the last node). This helps avoid local minima
    that pure Nearest Neighbor might fall into.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Get distances from current to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_node = -1
    best_score = np.inf
    
    # Iterate through each candidate node
    for i, u in enumerate(unvisited_nodes):
        # Cost to reach u from current
        cost_to_u = dists_from_current[i]
        
        # Find the nearest neighbor of u among the *other* unvisited nodes
        # If this is the second to last node, the next step will be to the last node.
        # If this is the last node, the next step is to destination_node.
        
        if n_unvisited == 2:
            # The other node is the only option
            other_idx = 1 - i
            v = unvisited_nodes[other_idx]
            cost_from_u = distance_matrix[u, v]
        else:
            # Get distances from u to all unvisited nodes
            dists_from_u = distance_matrix[u, unvisited_nodes]
            # Mask out u itself
            dists_from_u[i] = np.inf
            # Find the min distance
            min_dist_from_u = np.min(dists_from_u)
            # We don't strictly need the node id v, just the distance, 
            # but if we wanted to extend lookahead, we'd need it.
            cost_from_u = min_dist_from_u
        
        total_score = cost_to_u + cost_from_u
        
        if total_score < best_score:
            best_score = total_score
            best_node = u
            
    return int(best_node)

# EVOLVE-BLOCK-END
