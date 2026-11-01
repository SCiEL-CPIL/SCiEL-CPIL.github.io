"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import warnings

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return _covering_radius_brute(points)
    
    max_circ = 0.0
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        r1 = np.arccos(np.clip(n @ p1, -1, 1))
        r2 = np.arccos(np.clip(-n @ p1, -1, 1))
        r = min(r1, r2)
        max_circ = max(max_circ, r)
    
    return np.degrees(max_circ)


def _covering_radius_brute(points, n_theta=180, n_phi=360):
    """Brute force covering radius computation on a grid."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        g = grid[start:end]
        dots = g @ points.T
        dots = np.clip(dots, -1, 1)
        max_dots = np.max(dots, axis=1)
        min_angs = np.arccos(max_dots)
        max_dist = max(max_dist, np.max(min_angs))
    
    return np.degrees(max_dist)


def _lloyd_iteration(points, n_iter=50, tol=1e-10):
    """Lloyd's algorithm for spherical k-means / Voronoi centroid refinement."""
    points = np.array(points, dtype=float)
    n = len(points)
    
    for it in range(n_iter):
        # Compute Voronoi cells by finding nearest point for a dense grid
        # Use a moderately dense grid for efficiency
        n_theta = 360
        n_phi = 720
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Assign each grid point to nearest site
        dots = grid @ points.T  # (len(grid), n)
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroids of each Voronoi cell
        new_points = np.zeros_like(points)
        counts = np.zeros(n)
        for i in range(n):
            mask = nearest == i
            if np.sum(mask) > 0:
                new_points[i] = grid[mask].mean(axis=0)
                counts[i] = np.sum(mask)
        
        # Handle empty cells: keep original point
        for i in range(n):
            if counts[i] == 0:
                new_points[i] = points[i]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1)
        new_points = new_points / norms[:, None]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < tol:
            break
    
    return points


def _facet_polish(points, max_time=30.0):
    """Polish by iteratively moving worst-facet vertices toward their circumcenter."""
    points = np.array(points, dtype=float)
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.05
    no_improve = 0
    
    while time.time() - start < max_time and no_improve < 50:
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, None]
        
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            break
        
        # Find worst facet
        max_circ = 0.0
        worst_face = None
        for face in faces:
            p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            r1 = np.arccos(np.clip(n @ p1, -1, 1))
            r2 = np.arccos(np.clip(-n @ p1, -1, 1))
            r = min(r1, r2)
            if r > max_circ:
                max_circ = r
                worst_face = face
        
        if worst_face is None:
            break
        
        # Move vertices toward circumcenter
        p1, p2, p3 = points[worst_face[0]], points[worst_face[1]], points[worst_face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            break
        n = n / n_norm
        c = n if (n @ p1) > 0 else -n
        
        for idx in worst_face:
            p = points[idx]
            d = c - p
            d = d - (d @ p) * p
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-12:
                d = d / d_norm
                points[idx] = points[idx] + lr * d
        
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, None]
        
        new_radius = _covering_radius(points)
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            points = best_points.copy()
            lr *= 0.85
    
    return best_points


def _fibonacci_spiral(n):
    """Generate points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, None]


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Stage 1: Fibonacci spiral initialization
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Stage 2: Lloyd's algorithm for global balance
    if time.time() < deadline - 60:
        lloyd_points = _lloyd_iteration(points, n_iter=80)
        lloyd_radius = _covering_radius(lloyd_points)
        if lloyd_radius < best_radius:
            best_points = lloyd_points
            best_radius = lloyd_radius
    
    # Stage 3: Facet-based polishing
    if time.time() < deadline - 20:
        polished = _facet_polish(best_points, max_time=30)
        polish_radius = _covering_radius(polished)
        if polish_radius < best_radius:
            best_points = polished
            best_radius = polish_radius
    
    # Stage 4: Additional Lloyd + polish cycles if time permits
    remaining = deadline - time.time()
    if remaining > 25:
        lloyd2 = _lloyd_iteration(best_points, n_iter=40)
        lloyd2_radius = _covering_radius(lloyd2)
        if lloyd2_radius < best_radius:
            best_points = lloyd2
            best_radius = lloyd2_radius
        if time.time() < deadline - 15:
            polished2 = _facet_polish(best_points, max_time=min(15, deadline - time.time() - 5))
            p2_radius = _covering_radius(polished2)
            if p2_radius < best_radius:
                best_points = polished2
                best_radius = p2_radius
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, None]
    
    return best_points

# EVOLVE-BLOCK-END
