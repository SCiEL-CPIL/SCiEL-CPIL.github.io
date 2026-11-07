"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Combines a Delaunay-triangulation-based topological constraint (which
    encourages planar, non-crossing routes) with a power-transformed inverse
    distance term (which encourages short edges) and a Clarke-Wright savings
    component (which encourages merging routes and reducing depot returns).
    
    The heuristic is:
        eta[i, j] = (1 / d[i, j]^alpha) * (1 + beta * savings_norm[i, j]) * gate[i, j]
    
    where gate[i, j] is high for edges in the Delaunay triangulation or very
    close neighbors, and near-zero otherwise. This effectively restricts the
    search to a sparse geometrically meaningful graph.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Avoid division by zero; set diagonal and near-zero to large value
    d_safe = np.where(d > 1e-9, d, 1e9)
    
    # Power-transformed inverse distance: emphasizes short edges
    alpha = 2.0
    eta_base = 1.0 / (d_safe ** alpha)
    
    # Compute Clarke-Wright savings: s(i,j) = d(i,0) + d(0,j) - d(i,j)
    depot_dist = d[:, 0].copy()
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
    beta = 0.6
    eta = eta_base * (1.0 + beta * savings_norm)
    
    # Delaunay triangulation gate: encourage planar, non-crossing routes
    # Only consider customer coordinates (exclude depot at index 0)
    customer_coords = coordinates[1:]
    
    if len(customer_coords) >= 3:
        try:
            tri = Delaunay(customer_coords)
            # Create a gate matrix, initialized to near-zero
            gate = np.full((n, n), 1e-6)
            
            # Mark edges in the Delaunay triangulation as attractive
            for simplex in tri.simplices:
                # simplex contains 3 customer indices (0-based among customers)
                # Convert to global indices (add 1 for depot offset)
                i = simplex[0] + 1
                j = simplex[1] + 1
                k = simplex[2] + 1
                
                gate[i, j] = 1.0
                gate[j, i] = 1.0
                gate[i, k] = 1.0
                gate[k, i] = 1.0
                gate[j, k] = 1.0
                gate[k, j] = 1.0
            
            # Also allow edges from depot to any customer (necessary for route starts/ends)
            gate[0, :] = 1.0
            gate[:, 0] = 1.0
            
            # Additionally, allow very close neighbors not in the triangulation
            # (in case the triangulation misses some short edges)
            # Use a distance threshold: edges shorter than the 5th percentile of all pairwise distances
            dist_flat = d_safe[np.triu_indices(n, k=1)]
            if len(dist_flat) > 5:
                threshold = np.percentile(dist_flat, 15)
                close_mask = d_safe < threshold
                # Only apply to off-diagonal, non-depot edges not already gated
                np.fill_diagonal(close_mask, False)
                close_mask[0, :] = False
                close_mask[:, 0] = False
                gate[close_mask] = np.maximum(gate[close_mask], 0.5)
            
            # Apply the gate to the heuristic
            eta = eta * gate
            
        except Exception:
            # If Delaunay fails for any reason, fall back to no gating
            pass
    
    # Ensure all entries are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Diagonal should be very low (no self-loops)
    np.fill_diagonal(eta, 1e-9)
    
    # Safety: ensure we don't exceed reasonable time
    elapsed = time.time() - start_time
    if elapsed > 10:
        # If this took too long, return a simpler heuristic
        d_safe2 = np.where(d > 1e-9, d, 1e9)
        eta_simple = 1.0 / (d_safe2 ** 2.0)
        eta_simple = np.where(np.isfinite(eta_simple) & (eta_simple > 0), eta_simple, 1e-9)
        np.fill_diagonal(eta_simple, 1e-9)
        return eta_simple
    
    return eta
# EVOLVE-BLOCK-END
