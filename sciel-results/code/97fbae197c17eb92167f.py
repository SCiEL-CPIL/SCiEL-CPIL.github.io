"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius in radians by sampling the sphere densely."""
    # Use a Fibonacci sampling of the sphere to approximate the max distance
    M = 20000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute min distance from each sample to nearest point in pts
    # pts shape (N,3), samples shape (M,3)
    # Use chunked computation to save memory
    max_d = 0.0
    chunk = 2000
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        s = samples[start:end]  # (chunk, 3)
        # distances: (chunk, N)
        d = 1.0 - np.einsum('ij,kj->ik', s, pts)
        d = np.clip(d, -1.0, 1.0)
        min_d = np.min(d, axis=1)
        max_d = max(max_d, np.max(min_d))
    return np.arccos(np.clip(max_d, -1.0, 1.0))


def _local_refine(pts, max_iter=2000, tol=1e-10):
    """Local refinement: move each point to reduce local max distance."""
    pts = _normalize(pts.copy())
    prev_radius = _covering_radius(pts)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Find the "worst" direction: the point on the sphere farthest from pts[i] 
            # but within the Voronoi region. Approximate by moving toward centroid of 
            # nearest neighbors.
            # Compute distances from all points to pts[i]
            d = 1.0 - np.einsum('ij,j->i', pts, pts[i])
            d = np.clip(d, -1.0, 1.0)
            # Find k nearest neighbors
            k = min(10, N - 1)
            idx = np.argsort(d)[:k]
            # Move toward the centroid of these neighbors (in tangent space)
            neighbors = pts[idx]
            centroid = np.mean(neighbors, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid = centroid / centroid_norm
            
            # Project movement onto tangent plane at pts[i]
            v = centroid - np.dot(centroid, pts[i]) * pts[i]
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-14:
                continue
            
            # Try a few step sizes
            best_pts = pts.copy()
            best_rad = prev_radius
            for step in [0.01, 0.005, 0.002, 0.001]:
                new_pt = pts[i] + step * v
                new_pt = new_pt / np.linalg.norm(new_pt)
                trial = pts.copy()
                trial[i] = new_pt
                rad = _covering_radius(trial)
                if rad < best_rad - 1e-12:
                    best_rad = rad
                    best_pts = trial.copy()
            
            if best_rad < prev_radius - 1e-12:
                pts = best_pts
                prev_radius = best_rad
                improved = True
        
        if not improved:
            break
        if prev_radius < 0.32:  # ~18.3 degrees in radians
            break
    
    return pts, prev_radius


def construct():
    t_start = time.time()
    deadline = t_start + 100.0  # Safety margin before 120s
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_rad = _covering_radius(pts)
    
    # Local refinement
    pts, rad = _local_refine(pts)
    if rad < best_rad:
        best_pts = pts.copy()
        best_rad = rad
    
    # Check time
    if time.time() > deadline:
        return _normalize(best_pts)
    
    # Try a second refinement pass with different step sizes
    try:
        pts2, rad2 = _local_refine(best_pts, max_iter=500)
        if rad2 < best_rad:
            best_pts = pts2.copy()
            best_rad = rad2
    except Exception:
        pass
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
