"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ACO orienteering.
    
    The heuristic is designed to favor nodes with high prizes that are close to the origin (node 0).
    This is because the total path length is constrained, and nodes far from the origin are more expensive
    to include in a feasible tour that must return to the origin.
    
    eta[i, j] = (prize[j] + eps) / (dist[j, 0] + eps)
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    n = len(prizes)
    
    # Distance of each node from the origin (node 0)
    dist_from_origin = distance_matrix[:, 0]
    
    # Avoid division by zero
    eps = 1e-9
    
    # Heuristic for each target node j
    # High prize and low distance from origin -> High attractiveness
    node_attractiveness = (prizes + eps) / (dist_from_origin + eps)
    
    # The heuristic matrix is row-constant: eta[i, j] depends only on j
    # This encourages the colony to focus on the globally best nodes regardless of current position
    heuristic_matrix = np.tile(node_attractiveness, (n, 1))
    
    # Ensure non-negative and handle any potential issues (though logic above prevents 0/0)
    heuristic_matrix = np.maximum(heuristic_matrix, eps)
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
