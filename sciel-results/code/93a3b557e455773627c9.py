"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _spherical_circumcenter(v1, v2, v3):
    """
    Compute the circumcenter of a spherical triangle on the unit sphere.
    Returns (circumcenter, circumradius) where circumcenter is a unit vector.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate triangle, return centroid
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
    
    n = n / n_norm
    
    # Solve for circumcenter C = v1 + u*e1 + v*e2
    # such that C.v1 = C.v2 = C.v3
    A = np.array([
        [e1.dot(v1) - e1.dot(v2), e2.dot(v1) - e2.dot(v2)],
        [e1.dot(v1) - e1.dot(v3), e2.dot(v1) - e2.dot(v3)]
    ])
    b = np.array([
        v1.dot(v2) - 1.0,
        v1.dot(v3) - 1.0
    ])
    
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Degenerate, use centroid
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
        
    C = v1 + uv[0]*e1 + uv[1]*e2
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-10:
        return np.zeros(3), 0.0
    C = C / C_norm
    
    # Angular distance from C to v1
    dot = np.clip(np.dot(C, v1), -1.0, 1.0)
    r = np.arccos(dot)
    return C, r


def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # invalid, return large value
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        C, r = _spherical_circumcenter(verts[0], verts[1], verts[2])
        if r > max_radius:
            max_radius = r
    
    return max_radius


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm (Centroidal Voronoi on Sphere)
    def lloyd_step(points):
        try:
            hull = ConvexHull(points)
        except Exception:
            return points
            
        # Map each point to its adjacent facets
        point_facets = {i: [] for i in range(N)}
        for f in hull.simplices:
            for idx in f:
                point_facets[idx].append(f)
                
        new_points = np.zeros_like(points)
        for i in range(N):
            if not point_facets[i]:
                new_points[i] = points[i]
                continue
            centroid = np.zeros(3)
            for f in point_facets[i]:
                v1, v2, v3 = points[f[0]], points[f[1]], points[f[2]]
                C, _ = _spherical_circumcenter(v1, v2, v3)
                centroid += C
                
            if np.linalg.norm(centroid) > 1e-10:
                new_points[i] = centroid / np.linalg.norm(centroid)
            else:
                new_points[i] = points[i]
                
        return new_points

    # Run Lloyd's for a while
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        points = lloyd_step(points)

    # 3. Refinement: Reduce Covering Radius
    # Iterative refinement
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        max_r = 0.0
        max_facet_idx = -1
        max_circumcenter = None
        
        for idx, f in enumerate(hull.simplices):
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if r > max_r:
                max_r = r
                max_facet_idx = idx
                max_circumcenter = C
                
        if max_facet_idx == -1:
            break
            
        # Move the vertices of the worst facet towards its circumcenter
        # to shrink the facet and reduce its circumradius
        f = hull.simplices[max_facet_idx]
        step_size = 0.05 # Small step
        
        for idx in f:
            v = points[idx]
            # Move towards C
            new_v = v + step_size * (max_circumcenter - v)
            new_v = new_v / np.linalg.norm(new_v)
            points[idx] = new_v
        
        # Check if we improved
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # Revert if not improved
            points = best_points.copy()
            # Try smaller step
            step_size *= 0.5
            if step_size < 0.001:
                break

    return best_points
# EVOLVE-BLOCK-END
