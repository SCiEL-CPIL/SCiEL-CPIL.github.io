import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony orienteering.
    
    Structural change: Uses a slack-weighted prize-to-detour ratio.
    
    eta[i,j] = (prize[j] / (d[i,j] + d[j,0])) * max(0, max_len - (d[i,j] + d[j,0]))
    
    This formulation combines the efficiency of the move (prize per unit detour)
    with the remaining budget slack. It explicitly penalizes moves that consume
    most of the path budget, encouraging ants to save capacity for future
    high-value nodes, while still rewarding high prize density.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    eps = 1e-9
    
    # Calculate the "commitment" cost (detour) for each edge (i, j)
    # d[i, j] is shape (N, N)
    # d[j, 0] is shape (N,), broadcast along rows
    detour = distance_matrix + distance_matrix[:, 0][None, :]
    
    # Calculate the slack: how much budget remains if we take this detour
    # If detour > max_len, slack is 0 (infeasible or consumes all budget)
    slack = np.maximum(0.0, max_len - detour)
    
    # Calculate the prize-to-cost ratio
    # prize[j] is shape (N,), broadcast along columns
    prize_term = prizes + eps
    cost_term = detour + eps
    
    ratio = prize_term[None, :] / cost_term
    
    # Combine ratio and slack
    # This structural change weights the efficiency by the remaining budget
    heuristic = ratio * slack
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback heuristic
        fallback = np.ones_like(distance_matrix)
        return fallback
    
    # Ensure validity: non-positive or non-finite values become 1e-9
    # The ratio is positive. Slack is non-negative.
    # If slack is 0, heuristic is 0.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
