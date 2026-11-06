"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _spherical_circumcenter(v1, v2, v3):
    """
    Compute the circumcenter and circumradius of a spherical triangle on the unit sphere.
    Returns (circumcenter_unit_vector, circumradius_in_radians).
    """
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d

    A = np.array([
        [e1.dot(v2) - e1.dot(v1), e2.dot(v2) - e2.dot(v1)],
        [e1.dot(v3) - e1.dot(v1), e2.dot(v3) - e2.dot(v1)]
    ])
    b = np.array([
        1.0 - v1.dot(v2),
        1.0 - v1.dot(v3)
    ])

    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d

    C = v1 + uv[0]*e1 + uv[1]*e2
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-12:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
    C = C / C_norm

    dot = np.clip(np.dot(C, v1), -1.0, 1.0)
    r = np.arccos(dot)
    return C, r


def _compute_all_circumradii(points, hull):
    """Compute circumradii and centers for all facets."""
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for i in range(n_facets):
        f = simplices[i]
        C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
        radii[i] = r
        centers[i] = C
    
    return radii, centers


def _covering_radius(points):
    """Compute the covering radius (max circumradius) in radians."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    radii, _ = _compute_all_circumradii(points, hull)
    return np.max(radii)


def _lloyd_step(points):
    """One step of Lloyd's algorithm on the sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    point_facets = [[] for _ in range(N)]
    for f in hull.simplices:
        for idx in f:
            point_facets[idx].append(f)
    
    new_points = np.zeros_like(points)
    for i in range(N):
        if not point_facets[i]:
            new_points[i] = points[i]
            continue
        centroid = np.zeros(3)
        count = 0
        for f in point_facets[i]:
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            centroid += C
            count += 1
        if count > 0:
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    return new_points


def _refine_with_geodesic_bisectors(points, time_budget):
    """
    Refine the configuration by moving vertices of large facets towards their circumcenters.
    Uses adaptive step size and a smooth weighting scheme.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.1
    no_improve = 0
    patience = 20
    
    while time.time() - start < time_budget:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        radii, centers = _compute_all_circumradii(points, hull)
        max_r = np.max(radii)
        
        if max_r < best_radius - 1e-12:
            best_radius = max_r
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                break
        
        # Weight facets by their radius relative to max
        # Use a soft-max to focus on worst facets
        scale = 50.0
        w = np.exp(radii * scale - np.max(radii) * scale)
        w /= np.sum(w)
        
        # Select top K facets to update to save time
        # K can be dynamic, but let's take top 5-10
        top_k = 5
        idxs = np.argsort(-radii)[:top_k]
        
        displacements = np.zeros_like(points)
        for fi in idxs:
            f = hull.simplices[fi]
            C = centers[fi]
            wi = w[fi]
            for idx in f:
                v = points[idx]
                # Tangent vector from v to C
                proj = np.dot(v, C) * v
                tangent = C - proj
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent /= t_norm
                    displacements[idx] += wi * lr * tangent
        
        # Update points
        new_points = points + displacements
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= np.maximum(norms, 1e-15)
        
        # Check if new points are valid (no coincident points)
        # Simple check: min distance
        # This is expensive, so we skip unless necessary. 
        # The projection usually handles this well.
        
        points = new_points
        
        # Decay learning rate slowly
        lr *= 0.995
        if lr < 0.001:
            lr = 0.001
            
    return best_points, best_radius


def _micro_polish(points, time_budget):
    """
    Final high-precision polish. Iteratively moves the single worst facet's vertices
    towards its circumcenter with a very small, decaying step size.
    """
    start = time.time()
    current_points = points.copy()
    current_radius = _covering_radius(points)
    
    step = 0.01
    decay = 0.99
    
    while time.time() - start < time_budget:
        try:
            hull = ConvexHull(current_points)
        except Exception:
            break
            
        radii, centers = _compute_all_circumradii(current_points, hull)
        max_r = np.max(radii)
        
        if max_r < current_radius - 1e-10:
            current_radius = max_r
            # Keep step size or reset if stuck?
            # Let's just continue with decay
        else:
            # If we didn't improve, reduce step size faster
            step *= 0.9
            if step < 1e-6:
                break
                
        worst_idx = np.argmax(radii)
        f = hull.simplices[worst_idx]
        C = centers[worst_idx]
        
        new_points = current_points.copy()
        for idx in f:
            v = current_points[idx]
            # Slerp-like move
            new_v = v + step * (C - v)
            new_v /= np.linalg.norm(new_v)
            new_points[idx] = new_v
            
        current_points = new_points
        step *= decay
        
    return current_points, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm for uniform distribution
    # Allocate ~20s for Lloyd
    lloyd_deadline = start_time + 20.0
    for _ in range(200):
        if time.time() > lloyd_deadline:
            break
        points = _lloyd_step(points)
    
    # 3. Geodesic Bisector Refinement
    # Allocate ~60s for this
    refine_deadline = start_time + 80.0
    points, rad = _refine_with_geodesic_bisectors(points, refine_deadline - time.time())
    
    # 4. Micro Polish
    # Allocate remaining time minus safety buffer
    polish_time = deadline - time.time() - 5.0
    if polish_time > 0:
        points, rad = _micro_polish(points, polish_time)
    
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-15)
    
    return points
# EVOLVE-BLOCK-END
