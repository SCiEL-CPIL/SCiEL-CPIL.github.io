import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Returns an edge-attractiveness matrix based on a greedy ratio heuristic
    that emphasizes prize density relative to the remaining length budget.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Compute the cost of visiting node j from i and returning to depot
    dist_to_depot = distance_matrix[:, 0]
    round_trip_cost = distance_matrix + dist_to_depot[None, :]
    
    # Clip very small costs to avoid division issues
    round_trip_cost = np.where(round_trip_cost > 1e-12, round_trip_cost, 1e-12)
    
    # Compute the greedy ratio: prize at j divided by the round-trip cost
    numerator = prizes[None, :]
    raw_ratio = numerator / round_trip_cost
    
    # Identify valid entries for normalization
    valid_mask = (raw_ratio > 0) & np.isfinite(raw_ratio)
    
    if np.any(valid_mask):
        max_ratio = np.max(raw_ratio[valid_mask])
        if max_ratio > 0:
            normalized_ratio = raw_ratio / max_ratio
        else:
            normalized_ratio = np.zeros_like(raw_ratio)
    else:
        normalized_ratio = np.zeros_like(raw_ratio)
    
    # Apply a moderate power transformation to sharpen the heuristic
    # alpha = 2.5 provides a good balance between exploitation and exploration
    alpha = 2.5
    heuristic = normalized_ratio ** alpha
    
    # Ensure all entries are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
