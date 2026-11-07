import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix based on inverse distance.
    An edge (i, j) is attractive if the distance between i and j is small.
    
    H_ij = 1 / (d(i, j) + epsilon)
    """
    # Add a small epsilon to avoid division by zero for the diagonal and any identical coordinates
    # The diagonal will have a large value, but it is never selected as it is the current node (visited)
    eps = 1e-9
    heuristic = 1.0 / (distance_matrix + eps)
    
    # Ensure no non-finite values and positivity
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
