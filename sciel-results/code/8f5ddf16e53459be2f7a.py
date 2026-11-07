"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Args:
        prizes: Array of prizes for each node.
        distance_matrix: NxN matrix of distances between nodes.
        max_len: Maximum allowed length of the tour.
        
    Returns:
        Heuristic matrix of the same shape as distance_matrix.
    """
    n = len(prizes)
    # Create a copy to avoid modifying inputs if they are views
    d = distance_matrix.copy()
    
    # Ensure distances are positive to avoid division by zero or negative values
    # The problem states uniform random points, so distances are > 0 except d[i,i]=0.
    # We only care about i != j.
    
    # Heuristic: Prize of destination node divided by the sum of 
    # distance from current node to destination and distance from destination to depot (node 0).
    # This encourages visiting nodes that are both close to the current position 
    # and close to the depot, balancing exploration and return feasibility.
    
    # d[:, 0] is the distance of each node to the depot.
    # We want eta[i, j] = prizes[j] / (d[i, j] + d[j, 0])
    
    # Create a matrix where each column j is scaled by d[j, 0]
    # d_ret = d[:, 0] -> shape (N,)
    # We need to add d[j, 0] to each element in column j of d.
    # d + d_ret[None, :] adds d[j, 0] to every element in column j?
    # No, d_ret[None, :] has shape (1, N). 
    # d has shape (N, N).
    # d[i, j] + d_ret[j] is what we want.
    
    d_ret = d[:, 0]
    denom = d + d_ret[None, :]
    
    # Add epsilon to denominator to avoid division by zero (though d[i,i]=0 + d[i,0] might be small)
    # For i=j, d[i,i]=0. denom[i,i] = d[i,0]. If node i is 0, d[0,0]=0, denom=0.
    # We should handle the diagonal or just add epsilon.
    denom = denom + 1e-9
    
    # Numerator is prizes[j] for all i.
    num = prizes[None, :]
    
    eta = num / denom
    
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # Since prizes >= 0 and distances >= 0, eta >= 0.
    # If prize is 0, eta is 0. We replace 0 with 1e-9.
    # Note: Node 0 has prize 0. So eta[i, 0] will be 0.
    # The ants start at 0 and move to unvisited nodes. They don't move TO 0 during the loop 
    # (implied by "unvisited node j" and typically 0 is visited at start).
    # However, the matrix must be valid.
    
    eta[eta <= 0] = 1e-9
    
    # Ensure finite values
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
