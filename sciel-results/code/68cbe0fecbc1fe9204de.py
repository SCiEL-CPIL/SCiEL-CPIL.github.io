import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    # demands is a vector of floats, shape (n,). demands[0] is 0.
    # We want to penalize high demand nodes.
    # eta[i, j] = 1 / (distance[i, j] * (1 + demands[j]))
    
    # Ensure no division by zero or negative issues
    # distance_matrix[i, j] is > 0 for i != j. distance_matrix[i, i] is 0.
    # We only care about i != j moves.
    
    # Create a demand weight vector
    # demand_j = demands[j]
    # We want to scale the inverse distance by the inverse of (1 + demand_j)
    
    # demands shape (n,)
    # We need to broadcast demands to the columns of distance_matrix.
    # distance_matrix shape (n, n)
    # demands_col = demands.reshape(1, -1) -> shape (1, n)
    
    # eta[i, j] = 1.0 / (distance_matrix[i, j] * (1.0 + demands[j]))
    
    # Handle the diagonal (i=j). Distance is 0. 1/0 is inf.
    # The ACO loop picks unvisited customers. It won't pick i=j.
    # But the matrix must be finite.
    # The task says: "Heuristic entries that are not positive become 1e-9".
    # So if we produce inf, it becomes 1e-9? Or should we handle it?
    # "non-finite values ... make the whole result invalid".
    # So we MUST ensure all entries are finite.
    
    # Let's compute:
    # denom = distance_matrix * (1.0 + demands.reshape(1, -1))
    # eta = 1.0 / denom
    
    # For i=j, denom = 0 * (1+d) = 0. 1/0 = inf.
    # We should set diagonal to 0 or 1e-9?
    # The ant never picks a visited node. The depot is node 0.
    # If the ant is at 0, it picks j != 0.
    # If the ant is at j, it picks k != j.
    # So the diagonal values are never used in probability calculation?
    # "picks an unvisited customer ... with probability proportional to pheromone[i, j] * heuristic[i, j]"
    # Usually, the set of candidates excludes visited nodes and the current node.
    # However, to be safe against "non-finite values make the whole result invalid",
    # we should ensure the returned matrix has no inf/nan.
    
    # We can set diagonal to 0.
    # Or we can add a small epsilon to distance.
    # The baseline used `distance_matrix + 1e-9`.
    
    # Let's stick to the structure:
    demand_weights = 1.0 + demands.reshape(1, -1)
    denom = distance_matrix * demand_weights
    # Add epsilon to avoid division by zero on diagonal
    denom = denom + 1e-9
    eta = 1.0 / denom
    
    # Ensure non-positive becomes 1e-9?
    # "Heuristic entries that are not positive become 1e-9"
    # Since distances are positive and demands are positive, eta is positive.
    # So this rule handles any numerical edge cases.
    
    eta[eta <= 0] = 1e-9
    
    return eta
