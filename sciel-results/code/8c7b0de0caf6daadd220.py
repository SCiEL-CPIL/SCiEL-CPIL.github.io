"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic matrix.
    
    Uses a value-density heuristic:
    h[i,j] = prize[j] / (d[i,j] + d[j,0])^gamma
    
    This rewards high prizes and penalizes moves that consume a large
    fraction of the budget (since d[i,j] + d[j,0] is the total cost
    of visiting j and returning to start).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    # Parameters
    gamma = 1.0
    eps = 1e-9
    
    # d[j,0] is the cost to return to start from j
    d_j0 = distance_matrix[:, 0]
    
    # Total cost of visiting j from i and returning to 0: d[i,j] + d[j,0]
    # Note: d[i,j] is symmetric in Euclidean, but we use the matrix directly.
    # cost_matrix[i,j] = d[i,j] + d[j,0]
    cost_matrix = distance_matrix + d_j0[None, :]
    
    # Avoid division by zero
    cost_matrix = np.maximum(cost_matrix, eps)
    
    # Heuristic: prize[j] / cost[i,j]^gamma
    # prize_term is shape (n,), broadcast to (n, n)
    prize_term = np.maximum(prizes, eps)
    
    h = prize_term[None, :] / (cost_matrix ** gamma)
    
    # Ensure positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
