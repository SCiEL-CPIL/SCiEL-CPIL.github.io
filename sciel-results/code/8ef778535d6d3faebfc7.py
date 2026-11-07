"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    # Deadline guard (trivial for this function, but required by directive)
    start_time = time.time()
    deadline = start_time + 300 # 300s limit, use most of it if needed, but this is fast
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Check time, though unlikely to be hit
    if time.time() > deadline - 1:
         # Return a simple fallback if we are somehow stuck (should not happen)
         return np.ones((n, n)) * 1e-9

    d_j0 = distance_matrix[:, 0]  # (n,) return cost for each node
    
    # Net node value: prize minus a penalty for return distance
    return_penalty = 0.5
    net_value = prizes - return_penalty * d_j0
    
    eps = 1e-9
    net_value = np.maximum(net_value, eps)
    
    # Step cost
    step_cost = distance_matrix + eps
    
    # Base heuristic: net_value^alpha / step_cost^beta
    alpha = 2.0
    beta = 1.0
    
    prize_term = np.power(net_value[None, :], alpha)  # (1, n)
    cost_term = np.power(step_cost, beta)  # (n, n)
    
    base_heuristic = prize_term / cost_term
    
    # Slack weighting:
    # slack[i, j] = max_len - d[i, j] - d[j, 0]
    # If slack is positive, the move is "cheap" and leaves room.
    # We weight the heuristic by (1 + slack) to prefer moves that don't exhaust budget immediately.
    # This helps in finding combinations of nodes rather than one big greedy jump.
    
    # d[i, j] is distance_matrix
    # d[j, 0] is distance_matrix[:, 0] -> need to broadcast to (n, n)
    # d[j, 0] for each column j is distance_matrix[j, 0]
    # So d[j, 0] matrix is distance_matrix.T[0, :] broadcasted to rows?
    # distance_matrix is (n, n). distance_matrix[:, 0] is (n,).
    # We want d[i, j] + d[j, 0].
    # d[i, j] is distance_matrix[i, j].
    # d[j, 0] is distance_matrix[j, 0].
    # So we need to align them.
    
    # d_j0 is (n,).
    # We want a matrix where M[i, j] = d[j, 0].
    # This is d_j0[None, :] broadcast to (n, n).
    
    total_cost = distance_matrix + d_j0[None, :]
    
    slack = max_len - total_cost
    slack = np.maximum(slack, 0.0)
    
    # Weight factor: 1 + slack
    # If slack is 0, factor is 1.
    # If slack is 2, factor is 3.
    weight_factor = 1.0 + slack
    
    heuristic = base_heuristic * weight_factor
    
    # Row-normalize
    row_means = np.mean(heuristic, axis=1, keepdims=True)
    row_means = np.maximum(row_means, eps)
    heuristic = heuristic / row_means
    
    # Set diagonal to very small
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
