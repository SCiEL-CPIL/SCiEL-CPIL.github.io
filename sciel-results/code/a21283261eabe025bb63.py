import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    # Deadline guard (trivial for this computation, but included per directive)
    start_time = time.time()
    deadline = start_time + 300  # 300s limit
    
    # Convert inputs to numpy arrays for efficient vectorized operations
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure data validity: non-negative and finite
    # Clip to avoid negative values which might arise from numerical noise
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    # Handle non-finite values (NaN/Inf) by replacing with 0 or large numbers
    # If a distance is Inf, it should be treated as very unattractive (denom large)
    # If a prize is NaN, treat as 0
    prizes = np.where(np.isfinite(prizes), prizes, 0.0)
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e9)
    
    # Structural Change:
    # The seed heuristic was prize_j / d_ij. This ignores the cost of returning to the start.
    # In the Orienteering Problem, the constraint is L + d_ij + d_j0 <= max_len.
    # A better static heuristic accounts for the "total budget impact" of the move.
    # The move (i, j) costs d_ij in current path length and d_j0 in the final return leg.
    # Thus, the effective cost of visiting j from i is d_ij + d_j0.
    # We define attractiveness as prize divided by this effective cost.
    
    eps = 1e-9
    
    # d_j0 is the distance from node j to node 0 (start).
    # This is the 0th column of the distance matrix.
    return_dist = distance_matrix[:, 0]
    
    # Broadcast return_dist to align with distance_matrix
    # distance_matrix shape: (N, N) where index [i, j] is d_ij
    # return_dist shape: (N,) where index [j] is d_j0
    # We want denom[i, j] = d_ij + d_j0
    denom = distance_matrix + return_dist[None, :]
    
    # Numerator is prize_j, broadcast over all i
    numer = prizes[None, :]
    
    # Calculate heuristic
    heuristic = (numer + eps) / (denom + eps)
    
    # Ensure all entries are positive and finite
    # Non-positive or non-finite values become 1e-9 as per contract
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline (unlikely to trigger)
    if time.time() > deadline:
        # Return a valid, neutral heuristic if timeout occurs (though this shouldn't happen)
        return np.ones_like(distance_matrix) * 1e-9

    return heuristic
