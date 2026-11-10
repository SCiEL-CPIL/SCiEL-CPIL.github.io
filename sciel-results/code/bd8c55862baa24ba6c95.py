"""Hybrid heuristic for ACO CVRP: rank-based base with distance-exponent blend."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy:
    - Combine rank-based discrimination (smooth gradient, less sensitive to scale)
      with distance-based discrimination (stronger local signal) via a weighted blend.
    - The rank component provides a uniform gradient that allows pheromone to
      accumulate and guide search without premature convergence to local geometry.
    - The distance component (exponent 1.5) provides strong initial bias toward
      closer nodes, accelerating early exploration.
    - Blend weight is 0.6 for rank, 0.4 for distance, balancing exploration
      (rank) with exploitation (distance).
    - Normalization: scale so mean off-diagonal is ~1.0 to balance with
      initial pheromone of 1.0.
    """
    deadline = time.time() + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)

    eps = 1e-9

    # Component 1: Rank-based heuristic
    # Flatten and rank all pairwise distances globally
    flat_d = d.flatten()
    ranks = np.argsort(np.argsort(flat_d, kind='stable')) + 1
    rank_matrix = ranks.reshape(n, n)
    eta_rank = 1.0 / rank_matrix.astype(np.float64)

    # Component 2: Distance-based heuristic with exponent 1.5
    eta_dist = 1.0 / ((d + eps) ** 1.5)

    # Normalize each component to mean off-diagonal = 1.0 before blending
    mask = ~np.eye(n, dtype=bool)

    mean_rank = np.mean(eta_rank[mask])
    if mean_rank > 0 and np.isfinite(mean_rank):
        eta_rank = eta_rank / mean_rank

    mean_dist = np.mean(eta_dist[mask])
    if mean_dist > 0 and np.isfinite(mean_dist):
        eta_dist = eta_dist / mean_dist

    # Blend: 60% rank, 40% distance
    w_rank = 0.6
    w_dist = 0.4
    eta = w_rank * eta_rank + w_dist * eta_dist

    # Final normalization: ensure mean off-diagonal is ~1.0
    mean_eta = np.mean(eta[mask])
    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        # Fallback to simple inverse distance
        eta = 1.0 / (d + eps)
        mean_eta = np.mean(eta[mask])
        if mean_eta > 0 and np.isfinite(mean_eta):
            eta = eta / mean_eta

    # Sanitize: ensure positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9

    return eta
# EVOLVE-BLOCK-END
