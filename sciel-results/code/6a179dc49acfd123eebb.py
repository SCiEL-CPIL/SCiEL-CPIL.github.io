import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization using a local density of value model.
    
    The heuristic for moving from i to j is defined as:
    eta_ij = (Prize_j * Density_j) / (d_ij + d_j0)^2
    
    Where Density_j is the sum of prizes of the k-nearest neighbors of j, weighted by inverse distance.
    This rewards moving into high-value clusters.
    """
    start_time = time.time()
    n = len(prizes)
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Initialize density array
    density = np.zeros(n)
    
    # k-nearest neighbors for local density calculation
    k = 5
    
    # Calculate local density for each node j
    # To optimize, we can use matrix operations or simple loops since n=50 is small
    for j in range(n):
        # Get distances from j to all other nodes
        dists = distance_matrix[j].copy()
        dists[j] = np.inf # Exclude self
        
        # Get indices of k nearest neighbors
        # np.argpartition is faster than argsort for just getting k items
        if n > k:
            indices = np.argpartition(dists, k)[:k]
        else:
            indices = np.argsort(dists)[:n-1]
        
        # Calculate prize-weighted inverse distance sum
        # Weight = Prize_m / (d_jm + eps)
        eps = 1e-9
        neighbor_prizes = prizes[indices]
        neighbor_dists = dists[indices]
        
        # Avoid division by zero, though dists[j] is inf, others should be > 0
        weights = neighbor_prizes / (neighbor_dists + eps)
        density[j] = np.sum(weights)
    
    # Normalize density to prevent extreme values if prizes/dists vary wildly
    # We want the multiplier to be roughly O(1) or slightly > 1 for dense areas
    # Let's normalize by the mean density to keep scale stable
    mean_density = np.mean(density)
    if mean_density > 0:
        density = density / mean_density
    else:
        density = np.ones(n)
        
    # Construct heuristic matrix
    # eta_ij = (Prize_j * Density_j) / (d_ij + d_j0)^2
    
    # Prize_j and Density_j are vectors of length n
    # We need them as columns to broadcast against rows of distance_matrix
    # eta[i, j] depends on j's prize and density, and d[i, j] and d[j, 0]
    
    # d_j0 is the distance from j to 0, which is column 0 of distance_matrix?
    # distance_matrix[i, j] is d_ij. So d_j0 is distance_matrix[j, 0].
    d_j0 = distance_matrix[:, 0] # Vector of length n
    
    # Denominator: (d_ij + d_j0)^2
    # d_ij is (n, n)
    # d_j0 is (n,), needs to be (1, n) to broadcast across rows
    denom = (distance_matrix + d_j0[None, :]) ** 2
    
    # Numerator: Prize_j * Density_j
    # This is a vector of length n, needs to be (1, n)
    numer = (prizes * density)[None, :]
    
    # Calculate eta
    eta = numer / (denom + 1e-9)
    
    # Handle node 0 (start node). Usually, we don't move TO node 0 in the middle of the path
    # but the heuristic matrix is full. The ACO loop checks "unvisited node j".
    # The problem states "moves to an unvisited node j". Node 0 is visited at start.
    # So column 0 of eta will not be used for transitions *to* 0.
    # However, row 0 is used for the first move.
    # We should ensure eta[0, 0] is 0 or irrelevant.
    # The task says "Heuristic entries that are not positive become 1e-9".
    
    # Ensure positive values
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
