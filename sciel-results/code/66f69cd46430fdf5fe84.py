"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time


def heuristics(distance_matrix):
    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    
    n = dm.shape[0]
    start_time = time.time()
    deadline = 325.0  # Leave 5s margin before 330s kill

    # Extract points from distance matrix
    # For uniform random points in unit square, we can reconstruct approximate coordinates
    # using MDS (Multi-Dimensional Scaling) or just use the distance matrix directly
    # But for Delaunay, we need actual coordinates. Let's use classical MDS.
    
    # Classical MDS to recover 2D coordinates
    D2 = dm ** 2
    J = np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Take top 2 eigenvalues (sorted in descending order)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Coordinates: first 2 principal components
    coords = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Ensure coordinates are in a reasonable range (unit square)
    # Normalize to [0, 1] x [0, 1]
    if n > 1:
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1.0
        coords = (coords - min_coords) / range_coords
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback to inverse distance heuristic
        d = np.maximum(dm, 1e-9)
        h = 1.0 / (d ** 1.25)
        mean_h = np.mean(h)
        if not np.isfinite(mean_h) or mean_h <= 0.0:
            h = np.ones_like(dm, dtype=float)
        else:
            h = h / mean_h
        h = np.where(np.isfinite(h) & (h > 0.0), h, 1e-9)
        return h
    
    # Build adjacency from Delaunay triangulation
    # Edges in the triangulation get high attractiveness
    # Non-edges get low attractiveness (but still positive)
    
    # Initialize heuristic with low value
    h = np.full((n, n), 1e-6, dtype=float)
    h[np.diag_indices(n)] = 0.0  # Diagonal is 0 (will be handled by contract)
    
    # Mark Delaunay edges
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                # Attractiveness proportional to inverse distance
                dist = dm[u, v]
                if dist > 1e-9:
                    h[u, v] = 1.0 / dist
                    h[v, u] = 1.0 / dist
    
    # Also add a small amount of inverse distance for all edges to maintain connectivity
    # This ensures the graph remains connected and ants can still explore
    d = np.maximum(dm, 1e-9)
    base_h = 1.0 / (d ** 1.5)
    base_h /= np.mean(base_h)
    
    # Combine: Delaunay edges get boosted, others get base
    # Use a blend: 0.8 * Delaunay_boost + 0.2 * base
    # But only for off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    
    # For Delaunay edges, use the boosted value
    # For non-Delaunay edges, use base_h
    # We need to identify which edges are in the Delaunay triangulation
    
    # Create a boolean mask for Delaunay edges
    delaunay_mask = np.zeros((n, n), dtype=bool)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                delaunay_mask[u, v] = True
                delaunay_mask[v, u] = True
    
    # Final heuristic: Delaunay edges get 1/d, others get base_h
    h_final = np.where(delaunay_mask & mask, 1.0 / d, base_h)
    h_final[np.diag_indices(n)] = 0.0
    
    # Normalize to mean 1.0 for consistency
    mean_h = np.mean(h_final)
    if not np.isfinite(mean_h) or mean_h <= 0.0:
        h_final = np.ones_like(dm, dtype=float)
    else:
        h_final = h_final / mean_h
    
    # Contract requires positive entries; non-positive becomes 1e-9.
    h_final = np.where(np.isfinite(h_final) & (h_final > 0.0), h_final, 1e-9)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return simple fallback
        d = np.maximum(dm, 1e-9)
        h = 1.0 / (d ** 1.25)
        mean_h = np.mean(h)
        if not np.isfinite(mean_h) or mean_h <= 0.0:
            h = np.ones_like(dm, dtype=float)
        else:
            h = h / mean_h
        h = np.where(np.isfinite(h) & (h > 0.0), h, 1e-9)
        return h
    
    return h_final
# EVOLVE-BLOCK-END
