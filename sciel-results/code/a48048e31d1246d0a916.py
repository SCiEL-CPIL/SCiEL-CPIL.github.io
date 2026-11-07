"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Combines power-transformed inverse distance (encourages short edges)
    with Clarke-Wright savings (encourages merging routes, reducing depot
    returns). The savings component biases ants toward edges where
    connecting two customers directly is cheaper than routing both
    through the depot, which helps form fuller vehicles and reduces
    total number of routes.
    
    eta[i, j] = (1 / d[i,j]^alpha) * (1 + beta * savings_norm[i,j])
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Avoid division by zero; set diagonal and near-zero to large value
    d_safe = np.where(d > 1e-9, d, 1e9)
    
    # Power-transformed inverse distance: emphasizes short edges
    alpha = 2.0
    eta_base = 1.0 / (d_safe ** alpha)
    
    # Clarke-Wright savings: s(i,j) = d(i,0) + d(0,j) - d(i,j)
    # Positive when direct connection saves distance vs. going via depot
    depot_dist = d[:, 0].copy()  # d[i, 0] for all i
    savings = depot_dist[:, np.newaxis] + depot_dist[np.newaxis, :] - d
    
    # Only positive savings are meaningful; clamp negatives to 0
    savings_pos = np.maximum(savings, 0.0)
    
    # Normalize savings to [0, 1] for stable combination
    max_savings = savings_pos.max()
    if max_savings > 1e-9:
        savings_norm = savings_pos / max_savings
    else:
        savings_norm = np.zeros_like(savings_pos)
    
    # Combine: base inverse distance with savings bonus
    # beta controls the strength of the savings incentive
    beta = 0.8
    eta = eta_base * (1.0 + beta * savings_norm)
    
    # Ensure all entries are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Diagonal should be very low (no self-loops)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
