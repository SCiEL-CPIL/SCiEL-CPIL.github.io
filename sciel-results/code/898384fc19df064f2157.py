import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Returns an edge-attractiveness matrix based on a normalized, sharpened
    greedy ratio heuristic: (prizes[j] / (dist[i,j] + dist[j,0]))^alpha.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Compute denominator: cost to visit j and return to depot
    # dist[i,j] + dist[j,0]
    dist_to_depot = distance_matrix[:, 0]  # shape (n,)
    denominator = distance_matrix + dist_to_depot[None, :]  # shape (n, n)
    
    # Avoid division by zero
    denominator = np.where(denominator > 1e-12, denominator, 1e-12)
    
    # Compute raw ratio: prizes[j] / denominator
    # prizes[None, :] broadcasts to (n, n)
    numerator = prizes[None, :]
    raw_heuristic = numerator / denominator
    
    # Filter out invalid/zero values for normalization
    # We want to normalize based on the valid, positive entries
    valid_mask = (raw_heuristic > 0) & np.isfinite(raw_heuristic)
    
    if np.any(valid_mask):
        max_val = np.max(raw_heuristic[valid_mask])
        if max_val > 0:
            # Normalize to [0, 1]
            raw_heuristic = raw_heuristic / max_val
        else:
            raw_heuristic = np.zeros_like(raw_heuristic)
    else:
        raw_heuristic = np.zeros_like(raw_heuristic)
        
    # Apply power transformation to sharpen the heuristic
    # alpha = 2.0 emphasizes high-prize, low-cost moves
    alpha = 2.0
    heuristic = raw_heuristic ** alpha
    
    # Handle non-positive or non-finite values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
