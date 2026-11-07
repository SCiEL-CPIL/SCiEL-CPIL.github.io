import numpy as np

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    
    Uses inverse-cube distance as the heuristic. This provides a stronger
    bias towards short edges than the seed's inverse distance, which helps
    the ant colony find good tours more quickly in the limited number of
    iterations (30) available for small instances (N=50).
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Add a small epsilon to prevent division by zero (especially for the diagonal)
    # 1e-9 is a standard choice that aligns with the problem's invalid value threshold
    eps = 1e-9
    d_safe = d + eps
    
    # Inverse cube distance: sharper contrast between short and long edges
    # This mimics a stronger nearest-neighbor heuristic, which is effective
    # for the short run time and small problem size specified.
    eta = 1.0 / (d_safe ** 3)
    
    # Ensure all values are finite and positive
    # If any calculation resulted in inf or nan, replace with the minimum valid value
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
