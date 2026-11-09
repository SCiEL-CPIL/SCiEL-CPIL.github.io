import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    # distance_matrix is n x n
    # coordinates is n x 2
    
    # Initialize heuristic matrix
    eta = np.zeros((n, n))
    
    # Precompute distances from depot (node 0)
    d0 = distance_matrix[0, :]
    
    # For customer-customer edges (i, j where i, j > 0)
    # Use Savings heuristic: s_ij = d_0i + d_0j - d_ij
    # Heuristic value: s_ij / d_ij
    # This is high if the edge saves a lot of distance compared to going via depot,
    # and the edge itself is short.
    
    # Create indices for i, j > 0
    idx = np.arange(1, n)
    # Get submatrix for customers
    d_cust = distance_matrix[1:, 1:]
    d0_cust = d0[1:]
    
    # Calculate savings for all pairs of customers
    # s[i, j] = d0[i] + d0[j] - d[i, j]
    s = d0_cust[:, None] + d0_cust[None, :] - d_cust
    
    # Avoid division by zero, though d_ij should be > 0 for distinct nodes
    # If d_ij is 0 (same node), we don't care, but let's be safe
    d_safe = np.where(d_cust > 1e-9, d_cust, 1.0)
    
    # Heuristic for customer-customer
    eta_cust = s / d_safe
    
    # Ensure positivity
    eta_cust = np.maximum(eta_cust, 1e-9)
    
    # Fill the customer-customer part of eta
    eta[1:, 1:] = eta_cust
    
    # For depot edges (0, j) and (j, 0)
    # Use inverse distance
    # eta[0, j] = 1 / d[0, j]
    # eta[j, 0] = 1 / d[j, 0]
    
    d_depot = d0.copy()
    d_depot[0] = 1.0 # Avoid div by 0 for eta[0,0]
    
    eta_depot = 1.0 / d_depot
    eta_depot[0] = 0.0 # Depot to depot is not a valid move
    
    # Set row 0 and col 0
    eta[0, :] = eta_depot
    eta[:, 0] = eta_depot
    
    # Symmetry check (should be symmetric if distance matrix is)
    # eta = (eta + eta.T) / 2.0 # Optional, but good for consistency
    
    # Final check: replace non-positive with 1e-9
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
