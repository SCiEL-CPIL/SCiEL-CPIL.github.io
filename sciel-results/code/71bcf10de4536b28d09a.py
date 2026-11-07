import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony orienteering.
    
    Uses a sharp ratio of prize to marginal detour cost.
    eta[i,j] = (prize[j] / (d[i,j] + d[j,0]))^k
    
    This formulation directly rewards nodes that offer high prize
    relative to the "commitment" required to visit them and return.
    The power k sharpens the preference for high-efficiency moves,
    guiding ants to high-value, low-cost targets early in the path.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure inputs are valid
    if prizes.size == 0 or distance_matrix.size == 0:
        return np.ones((0,0))
        
    n = len(prizes)
    
    # Handle potential zero distances or invalid entries
    # d[j, 0] is the cost to return to start from j
    # d[i, j] is the cost to go from i to j
    # Marginal cost = d[i, j] + d[j, 0]
    
    # d[j, 0] shape: (N, 1) to broadcast across rows i
    # Actually, distance_matrix is (N, N).
    # dist_to_start = distance_matrix[:, 0] -> shape (N,)
    # We need d[j, 0] for each column j.
    # distance_matrix[:, 0] gives d[i, 0] for all i.
    # We want d[j, 0]. This is the same vector, just indexed by column j.
    # So, cost_return = distance_matrix[:, 0] -> shape (N,)
    # We need to add this to each row of distance_matrix?
    # No.
    # detour[i, j] = distance_matrix[i, j] + distance_matrix[j, 0]
    # distance_matrix[i, j] is (N, N)
    # distance_matrix[j, 0] is a vector of length N, where index j is the value.
    # So we need to broadcast distance_matrix[:, 0] along the columns (axis 1).
    # dist_to_start = distance_matrix[:, 0] # shape (N,)
    # dist_to_start[None, :] # shape (1, N)
    
    dist_to_start = distance_matrix[:, 0]
    # detour[i, j] = distance_matrix[i, j] + dist_to_start[j]
    detour = distance_matrix + dist_to_start[None, :]
    
    # Avoid division by zero
    eps = 1e-9
    safe_detour = np.maximum(detour, eps)
    
    # Prize term
    # prize[j] is shape (N,), broadcast along rows (axis 0)
    prize_term = prizes[None, :] + eps
    
    # Ratio
    ratio = prize_term / safe_detour
    
    # Power
    k = 3.0
    heuristic = ratio ** k
    
    # Clean up non-finite or non-positive values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
