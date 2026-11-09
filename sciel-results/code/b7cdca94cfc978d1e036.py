import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    # Convert to numpy arrays if not already
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # d_j0 is the distance from each node j to the depot (node 0)
    d_j0 = distance_matrix[:, 0]
    
    # Denominator: distance from i to j plus distance from j to depot
    # This approximates the total length increase if we go i->j->0
    denom = distance_matrix + d_j0[None, :]
    
    # Numerator: prize of destination j
    numer = prizes[None, :]
    
    # Calculate heuristic
    # Add small epsilon to denominator to avoid division by zero
    H = numer / (denom + 1e-9)
    
    # Enforce positivity: entries that are not positive become 1e-9
    H = np.where(H > 0, H, 1e-9)
    
    return H
