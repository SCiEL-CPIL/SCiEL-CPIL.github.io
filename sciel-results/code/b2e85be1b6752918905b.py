"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for CVRP using a 2-opt informed heuristic.
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    # Base heuristic: Inverse distance with sharpened power
    # Avoid division by zero
    d = distance_matrix + 1e-9
    eta = 1.0 / (d ** 1.5)
    
    # Neighborhood Consistency Bonus
    # For each node, find its 2 nearest neighbors
    # This helps identify edges that are part of the "skeleton" of the optimal tour
    
    # Get indices of 2 smallest distances for each row
    # np.argpartition is faster than full sort
    # We need the indices of the 2 smallest values in each row
    
    # Create a mask to ignore self-loops (distance 0)
    # distance_matrix[i][i] is 0, so it will always be the smallest.
    # We need the 2 smallest excluding the diagonal.
    
    # Set diagonal to inf for partitioning
    d_for_nn = distance_matrix.copy()
    np.fill_diagonal(d_for_nn, np.inf)
    
    # Get the indices of the 2 smallest distances for each node
    # shape: (n, 2)
    try:
        # k=2 smallest
        nn_indices = np.argpartition(d_for_nn, 2, axis=1)[:, :2]
    except Exception:
        # Fallback if n < 3
        nn_indices = np.zeros((n, 2), dtype=int)
        for i in range(n):
            if n > 1:
                nn_indices[i] = np.argsort(d_for_nn[i])[:min(2, n-1)]
            else:
                nn_indices[i] = [0, 0]

    # Create a bonus matrix
    bonus = np.ones_like(eta)
    
    # Check mutual nearest neighbors
    # If j is in NN(i) and i is in NN(j), it's a strong edge
    
    for i in range(n):
        n1 = nn_indices[i, 0]
        n2 = nn_indices[i, 1]
        
        # Check if i is in NN(n1)
        if i in nn_indices[n1]:
            bonus[i, n1] *= 2.0
            bonus[n1, i] *= 2.0 # Symmetric heuristic usually, but ACO uses directed? 
            # The task says "edge-attractiveness matrix", usually symmetric in TSP/CVRP spatial heuristics.
            # But pheromone is symmetric? "pheromone[i, j]".
            # Let's assume symmetric heuristic is fine, or we can make it directed.
            # ACO usually uses directed pheromone, but heuristic can be symmetric.
            # The prompt says "returns an edge-attractiveness matrix ... of the same shape".
            # Let's keep it symmetric for simplicity and robustness.
            
        if i in nn_indices[n2]:
            bonus[i, n2] *= 2.0
            bonus[n2, i] *= 2.0

    # Apply bonus
    eta = eta * bonus
    
    # Ensure non-finite values are handled
    eta[~np.isfinite(eta)] = 1e-9
    
    # Ensure positive
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
