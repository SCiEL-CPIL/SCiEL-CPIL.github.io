import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization.
    
    The heuristic is designed to balance the prize value of a node against
    the cost of visiting it, considering both the distance from the current
    node and the distance back to the origin.
    
    Formula:
    eta[i, j] = (prize[j] + eps)^alpha / ((d[i, j] + eps)^beta * (d[j, 0] + eps)^gamma)
    
    This multiplicative penalty in the denominator strongly favors nodes that are
    both close to the current position and close to the origin, preserving
    path length budget.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Parameters
    alpha = 3.0
    beta = 1.5
    gamma = 1.0
    eps = 1e-9
    
    # Extract distance to origin for each node (column 0 or row 0 depending on symmetry, 
    # but distance_matrix[j, 0] is the distance from j to 0)
    # Note: distance_matrix is symmetric, so d[j, 0] == d[0, j]
    dist_to_origin = distance_matrix[:, 0]
    
    # Calculate heuristic
    # prize_term: shape (N,) -> broadcast to (N, N) along axis 0
    prize_term = (prizes + eps) ** alpha
    
    # dist_ij_term: shape (N, N)
    dist_ij_term = (distance_matrix + eps) ** beta
    
    # dist_j0_term: shape (N,) -> broadcast to (N, N) along axis 0 (rows are i, cols are j)
    # We want dist_to_origin[j] for each column j
    dist_j0_term = (dist_to_origin + eps) ** gamma
    # Broadcast dist_j0_term to match distance_matrix shape
    # dist_j0_term is (N,), we need it to be (1, N) to broadcast across rows
    dist_j0_term = dist_j0_term[None, :]
    
    heuristic = prize_term[None, :] / (dist_ij_term * dist_j0_term)
    
    # Ensure validity: positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
