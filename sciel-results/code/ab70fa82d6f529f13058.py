import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    # Initialize heuristic matrix
    eta = np.zeros((n, n))
    
    # Use pure inverse distance for all off-diagonal entries
    # This encourages the ant to move to the nearest feasible neighbor.
    # The feasibility check in the ACO loop will handle capacity constraints.
    # The pheromone will learn which "nearest" choices lead to good global solutions.
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Add a small epsilon to avoid division by zero (though distances should be > 0 for distinct nodes in unit square)
                eta[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
            else:
                eta[i, j] = 0.0
                
    return eta
