"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Parameters
    alpha = 3.0
    beta = 1.5
    gamma = 1.0
    eps = 1e-9
    
    # Extract distance to origin for each node
    dist_to_origin = distance_matrix[:, 0]
    
    # Calculate heuristic components
    # prize_term: shape (N,) -> broadcast to (N, N) along axis 0
    prize_term = (prizes + eps) ** alpha
    
    # dist_ij_term: shape (N, N)
    dist_ij_term = (distance_matrix + eps) ** beta
    
    # dist_j0_term: shape (N,) -> broadcast to (N, N) along axis 0 (rows are i, cols are j)
    dist_j0_term = (dist_to_origin + eps) ** gamma
    dist_j0_term = dist_j0_term[None, :]
    
    # Base heuristic: prize^alpha / (d_ij^beta * d_j0^gamma)
    base_heuristic = prize_term[None, :] / (dist_ij_term * dist_j0_term)
    
    # Marginal gain component: weight by remaining budget slack
    # Slack for edge (i, j) = max_len - d[i,j] - d[j,0]
    # Only positive slack means the edge is feasible and has room for more
    slack = max_len - distance_matrix - dist_to_origin[np.newaxis, :]
    slack = np.maximum(slack, 0.0)
    
    # Normalize slack to [0, 1] for weighting
    max_slack = np.max(slack)
    if max_slack > 0:
        slack_normalized = slack / max_slack
    else:
        slack_normalized = np.ones_like(slack)
    
    # Combine: use a smooth blend of base heuristic and slack-weighted version
    # The slack factor emphasizes edges that leave more budget for future nodes
    # Use a multiplicative factor that boosts edges with more slack
    slack_factor = 1.0 + slack_normalized
    
    heuristic = base_heuristic * slack_factor
    
    # Ensure validity: positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
