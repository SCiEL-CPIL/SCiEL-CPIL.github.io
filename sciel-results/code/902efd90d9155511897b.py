import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix based on the savings principle (Clarke-Wright).
    An edge (i, j) is attractive if linking i and j saves distance compared to
    serving them separately from the depot.
    
    H_ij = scale * (d(i, depot) + d(j, depot) - d(i, j))
    """
    # Get distances from all nodes to the depot (node 0)
    d_depot = distance_matrix[:, 0]
    
    # Calculate savings matrix using broadcasting
    # d_depot is shape (N, 1)
    # d_depot.T is shape (1, N)
    # distance_matrix is shape (N, N)
    savings = d_depot[:, np.newaxis] + d_depot[np.newaxis, :] - distance_matrix
    
    # Ensure values are positive (triangle inequality ensures >= 0, but float precision might dip below)
    savings = np.maximum(savings, 1e-9)
    
    # Scale the heuristic to be in a similar range as 1/distance
    # 1/distance is typically between 1 and 20 for unit square.
    # Savings are typically between 0 and 2.
    # Multiplying by 10 brings them to a comparable magnitude.
    heuristic = savings * 10.0
    
    return heuristic
