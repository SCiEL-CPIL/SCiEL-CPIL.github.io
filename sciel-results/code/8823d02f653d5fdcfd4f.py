"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony optimization of orienteering.
    
    Combines:
    1. Feasibility masking (edge must be part of a valid tour)
    2. Dual-distance penalty: prize[j] / (d[i,j]^alpha * d[j,0]^gamma)
    3. Slack boost: edges leaving more budget room get a multiplier
    """
    start_time = time.time()
    DEADLINE = start_time + 290.0  # 300s limit, keep 10s margin

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    # Ensure shape correctness
    if distance_matrix.shape != (n, n):
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default

    eps = 1e-9

    # Distance from each node back to origin
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    min_dist_from_0 = distance_matrix[0, :]  # shape (n,)

    # For each edge (i, j), estimate minimum tour length:
    # min_dist_from_0[i] + d[i,j] + d[j,0]
    min_tour_len = min_dist_from_0[:, None] + distance_matrix + d_j0[None, :]

    # Feasibility mask: edge (i,j) can be part of a valid tour
    feasible = min_tour_len <= max_len

    # Slack: how much budget remains after taking this edge (minimum case)
    slack = max_len - min_tour_len
    slack = np.maximum(slack, 0.0)

    # Heuristic: prize[j] / (d[i,j]^alpha * d[j,0]^gamma)
    # With 50 points in unit square and max_len=3, distances are ~0.1-1.4
    # alpha=1.0, gamma=0.8 balances approach cost vs return cost
    # Prize is in [0,1], so we add eps to avoid zero
    alpha = 1.0
    gamma = 0.8

    safe_d_ij = distance_matrix + eps
    safe_d_j0 = d_j0[None, :] + eps

    # Compute the core heuristic
    prize_term = prizes[None, :] + eps  # (1, n)
    d_ij_term = safe_d_ij ** alpha       # (n, n)
    d_j0_term = safe_d_j0 ** gamma       # (1, n)

    base_heuristic = prize_term / (d_ij_term * d_j0_term)

    # Apply feasibility: non-feasible edges get eps
    heuristic = np.where(feasible, base_heuristic, eps)

    # Boost edges with more slack (they are more flexible)
    # Use a moderate boost: multiply by (1 + slack / max_len)^0.5
    boost = 1.0 + slack / max_len
    boost = np.power(boost, 0.5)
    boost = np.where(feasible, boost, 1.0)
    heuristic = heuristic * boost

    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)

    # Set diagonal to eps (self-loops not useful)
    np.fill_diagonal(heuristic, eps)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        pass  # Return what we have

    return heuristic
# EVOLVE-BLOCK-END
