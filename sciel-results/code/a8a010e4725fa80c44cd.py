"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Strategy:
    - Base: power-transformed inverse distance (alpha=1.5)
    - Savings bonus: Clarke-Wright savings normalized and added multiplicatively
    - Geometric gate: Delaunay + K-NN (k=8) + distance percentile (20th)
    - Depot connections always allowed
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Safety deadline check
    if time.time() - start_time > 20:
        d_safe = np.where(d > 1e-9, d, 1e9)
        eta = 1.0 / (d_safe ** 1.5)
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        np.fill_diagonal(eta, 1e-9)
        return eta
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-9, d, 1e9)
    
    # Power-transformed inverse distance
    alpha = 1.5
    eta_base = 1.0 / (d_safe ** alpha)
    
    # Clarke-Wright savings
    depot_dist = d[:, 0].copy()
    savings = depot_dist[:, np.newaxis] + depot_dist[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    
    max_savings = savings_pos.max()
    if max_savings > 1e-9:
        savings_norm = savings_pos / max_savings
    else:
        savings_norm = np.zeros_like(savings_pos)
    
    # Combine base with savings
    beta = 1.0
    eta = eta_base * (1.0 + beta * savings_norm)
    
    # Hybrid geometric gate: Delaunay + K-NN + distance percentile
    customer_coords = coordinates[1:]
    gate = np.full((n, n), 1e-6)
    
    # 1. Delaunay triangulation edges (topological, non-crossing)
    if len(customer_coords) >= 3:
        try:
            tri = Delaunay(customer_coords)
            for simplex in tri.simplices:
                i = simplex[0] + 1
                j = simplex[1] + 1
                k = simplex[2] + 1
                gate[i, j] = 1.0
                gate[j, i] = 1.0
                gate[i, k] = 1.0
                gate[k, i] = 1.0
                gate[j, k] = 1.0
                gate[k, j] = 1.0
        except Exception:
            pass
    
    # 2. K-NN edges (local proximity, k=8)
    k = 8
    if len(customer_coords) >= 3:
        try:
            tree = cKDTree(customer_coords)
            for i in range(1, n):
                dists, indices = tree.query(customer_coords[i - 1], k=min(k, len(customer_coords) - 1))
                for j_idx in indices:
                    j = j_idx + 1
                    if j != i:
                        gate[i, j] = 1.0
                        gate[j, i] = 1.0
        except Exception:
            pass
    
    # 3. Distance percentile fallback (edges shorter than 20th percentile)
    try:
        dist_flat = d_safe[np.triu_indices(n, k=1)]
        if len(dist_flat) > 5:
            threshold = np.percentile(dist_flat, 20)
            close_mask = d_safe < threshold
            np.fill_diagonal(close_mask, False)
            close_mask[0, :] = False
            close_mask[:, 0] = False
            gate[close_mask] = np.maximum(gate[close_mask], 0.5)
    except Exception:
        pass
    
    # Allow depot to connect to all customers
    gate[0, :] = 1.0
    gate[:, 0] = 1.0
    
    # Apply gate
    eta = eta * gate
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    np.fill_diagonal(eta, 1e-9)
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > 20:
        d_safe2 = np.where(d > 1e-9, d, 1e9)
        eta_simple = 1.0 / (d_safe2 ** 1.5)
        eta_simple = np.where(np.isfinite(eta_simple) & (eta_simple > 0), eta_simple, 1e-9)
        np.fill_diagonal(eta_simple, 1e-9)
        return eta_simple
    
    return eta
# EVOLVE-BLOCK-END
