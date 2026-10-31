"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a dynamic weighted heuristic.
    
    The score for each candidate node u is:
        score(u) = dist(current, u) + alpha(k) * dist(u, destination) + beta * avg_dist(u, unvisited)
        
    Where:
    - alpha(k) is a dynamic weight that decreases as the number of unvisited nodes k decreases.
      This ensures that early in the tour, we steer towards the destination to prevent
      getting stuck in a corner, while late in the tour, we focus on local optimality
      (nearest neighbor) to minimize the final steps.
    - beta is a small constant weight for the average distance to other unvisited nodes,
      which acts as a regularization term to avoid sparse areas.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate average distance to other unvisited nodes for each candidate
    # This is O(n^2) but n is small (<=50)
    unvisited_idx = unvisited_nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # Set diagonal to 0 for averaging
    row_sums = np.sum(submatrix, axis=1)
    avg_dist_to_other = row_sums / (n_unvisited - 1)
    
    # Dynamic alpha: starts high, ends low
    # Assume total nodes is around 50. 
    # We use a linear decay based on the fraction of nodes remaining.
    # If we don't know N exactly, we can assume 50 based on the problem description.
    # To be safe, we can just use n_unvisited directly if we assume it starts at 50.
    # Let's use a fixed reference N=50.
    N_REF = 50
    alpha = n_unvisited / N_REF
    if alpha > 1.0:
        alpha = 1.0
    elif alpha < 0.0:
        alpha = 0.0
        
    # Beta for the average distance penalty
    beta = 0.2
    
    # Compute the combined score
    scores = dist_from_current + alpha * dist_to_dest + beta * avg_dist_to_other
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
