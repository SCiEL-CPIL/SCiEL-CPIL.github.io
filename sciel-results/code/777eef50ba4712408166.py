import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    # distance_matrix is n x n
    # coordinates is n x 2
    # demands is n,
    
    # Initialize heuristic with inverse distance raised to a power for sharper discrimination
    # Add small epsilon to avoid division by zero
    d = distance_matrix + 1e-9
    # Use exponent 1.5 to make shorter edges much more attractive
    h = 1.0 / (d ** 1.5)
    
    # Compute savings
    # s_ij = d_i0 + d_0j - d_ij
    d_i0 = distance_matrix[:, 0]
    d_0j = distance_matrix[0, :]
    
    # s is n x n
    s = d_i0[:, np.newaxis] + d_0j[np.newaxis, :] - distance_matrix
    
    # Strategy: h_ij = (1/d_ij^1.5) * (1 + alpha * s_ij)
    # The sharper inverse distance gradient combined with savings boost
    # should help ants converge faster to shorter routes
    
    # Use alpha = 1.0 for savings boost
    h = h * (1.0 + s)
    
    # Ensure non-finite and positive
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    return h
