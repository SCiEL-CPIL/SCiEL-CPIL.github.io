"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Combines a power-transformed inverse distance (to encourage short edges
    and tight local clusters) with a Clarke-Wright savings component (to
    encourage merging routes and reducing depot returns).
    
    The heuristic is:
        eta[i, j] = (1 / d[i, j]^alpha) * (1 + beta * savings[i, j]_norm)
    
    where savings[i, j] = d[i, 0] + d[0, j] - d[i, j]
    and savings is normalized to [0, 1] range for stability.
    
    Parameters:
        distance_matrix: (n, n) matrix of pairwise distances
        coordinates: (n, 2) array of node coordinates
        demands: (n,) array of node demands
        capacity: vehicle capacity
    
    Returns:
        (n, n) matrix of heuristic attractiveness (larger = more attractive)
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Avoid division by zero; diagonal is 0
    d_safe = np.where(d > 1e-9, d, 1e9)
    
    # Power-transformed inverse distance: emphasizes short edges
    alpha = 2.0
    eta_base = 1.0 / (d_safe ** alpha)
    
    # Compute Clarke-Wright savings: s(i,j) = d(i,0) + d(0,j) - d(i,j)
    # This is positive when connecting i and j directly is better than
    # going through the depot
    depot_dist = d[:, 0]  # d[i, 0] for all i
    savings = depot_dist[:, np.newaxis] + depot_dist[np.newaxis, :] - d
    
    # Normalize savings to [0, 1] for stability
    # Only consider positive savings; negative savings get 0
    savings_pos = np.maximum(savings, 0.0)
    max_savings = savings_pos.max()
    if max_savings > 1e-9:
        savings_norm = savings_pos / max_savings
    else:
        savings_norm = np.zeros_like(savings_pos)
    
    # Combine: base inverse distance with savings bonus
    # beta controls how much savings influences the heuristic
    beta = 0.5
    eta = eta_base * (1.0 + beta * savings_norm)
    
    # Ensure all entries are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Diagonal should be very low (no self-loops)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
