"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Ensure the matrix is square and valid
    if d.shape[0] != d.shape[1]:
        return np.ones_like(d)
        
    # Work on a copy to avoid modifying the original
    d_safe = d.copy()
    
    # Set diagonal to a large value so nodes don't pick themselves
    # The prompt says non-positive becomes 1e-9, but diagonal is 0.
    # We want diagonal to be non-attractive.
    np.fill_diagonal(d_safe, np.inf)
    
    # Parameters for the heuristic
    K = 12       # Number of nearest neighbors to keep
    alpha = 3.0  # Exponent for distance decay
    
    # Initialize heuristic with a very small value
    eta = np.full_like(d_safe, 1e-9)
    
    # For each node, find the K nearest neighbors and assign high attractiveness
    for i in range(n):
        # Get indices of K smallest distances from node i
        # argsort returns indices that would sort the array
        indices = np.argsort(d_safe[i])[:K]
        
        for j in indices:
            if d_safe[i, j] > 0 and np.isfinite(d_safe[i, j]):
                eta[i, j] = 1.0 / (d_safe[i, j] ** alpha)
            else:
                eta[i, j] = 1e-9
                
    # Symmetrize the heuristic matrix.
    # If i considers j a neighbor, or j considers i a neighbor, the edge should be attractive.
    # Using max ensures that the "strongest" local connection dictates the weight.
    eta = np.maximum(eta, eta.T)
    
    # Final safety check: replace any non-finite or non-positive values with 1e-9
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
