import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    # Ensure types are float
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # d_j0 is the distance from each node j to node 0
    d_j0 = distance_matrix[:, 0]
    
    # We want eta[i, j] proportional to p_j / (d_ij + d_j0)
    # d_ij is distance_matrix[i, j]
    # d_j0 is a vector of shape (N,), needs to be broadcast to (N, N)
    # distance_matrix is (N, N). d_j0[None, :] is (1, N).
    # d_ij + d_j0[None, :] will be (N, N) where element (i, j) is d_ij + d_j0.
    
    denom = distance_matrix + d_j0[None, :]
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    denom = denom + eps
    
    # Numerator is prize of destination j
    # prizes[None, :] is (1, N)
    numer = prizes[None, :] + eps
    
    H = numer / denom
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, eps)
    
    # Ensure positive
    H = np.maximum(H, eps)
    
    return H
