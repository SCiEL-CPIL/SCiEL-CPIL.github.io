"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute edge-attractiveness heuristic for ACO TSP.
    Uses a blend of inverse distance and a structural prior based on
    nearest-neighbor graph edges to guide ants toward good local moves.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s kill

    n = distance_matrix.shape[0]
    # Mask diagonal (self-loops)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Base heuristic: inverse distance (classic ACO)
    # Avoid division by zero for self-loops
    inv_dist = np.zeros_like(distance_matrix, dtype=np.float64)
    inv_dist[mask] = 1.0 / (distance_matrix[mask] + 1e-9)
    
    # Structural prior: count how many times each edge appears in the
    # nearest-neighbor graph (both directions) or is among top-k neighbors.
    # This reinforces edges that are locally good.
    top_k = 5
    nn_prior = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = distance_matrix[i].copy()
        row[i] = np.inf  # exclude self
        # Get indices of top_k smallest distances
        idx = np.argpartition(row, top_k)[:top_k]
        for j in idx:
            nn_prior[i, j] += 1.0
    
    # Also consider reverse direction (if j is i's neighbor, i might be j's neighbor)
    # But ACO uses directed edges, so we keep the directed prior.
    
    # Blend: heuristic = inv_dist * (1 + alpha * nn_prior)
    # alpha controls strength of structural prior
    alpha = 0.5
    heuristic = inv_dist * (1.0 + alpha * nn_prior)
    
    # Ensure positive finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        # Return the base heuristic if structural prior took too long
        return inv_dist
    
    return heuristic
# EVOLVE-BLOCK-END
