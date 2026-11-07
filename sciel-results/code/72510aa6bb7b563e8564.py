import numpy as np

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for ACO TSP.
    
    Uses inverse distance raised to a tuned power, normalized so that
    the mean of off-diagonal entries is 1. This biases the fixed ACO
    toward shorter edges without requiring expensive internal search.
    """
    n = len(distance_matrix)
    
    # Use a moderate exponent that balances exploration vs exploitation
    # alpha=1.25 was found to improve over alpha=1.0 in prior measurements
    alpha = 1.25
    
    # Create safe distance matrix (avoid division by zero on diagonal)
    safe_dist = distance_matrix.copy().astype(np.float64)
    np.fill_diagonal(safe_dist, 1.0)
    
    # Compute heuristic: eta[i,j] = 1 / d[i,j]^alpha
    eta = 1.0 / (safe_dist ** alpha)
    
    # Normalize so mean of off-diagonal entries is 1
    # This makes the heuristic scale-invariant and comparable across instances
    mask = ~np.eye(n, dtype=bool)
    mean_offdiag = np.mean(eta[mask])
    if mean_offdiag > 0:
        eta = eta / mean_offdiag
    
    # Ensure diagonal is positive (will be masked in ACO, but keep valid)
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all entries are positive
    eta[eta <= 0] = 1e-9
    
    return eta
