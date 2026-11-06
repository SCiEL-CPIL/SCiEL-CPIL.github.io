"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize_scalar


def _spherical_circumradius(a, b, c):
    """Compute the spherical circumradius and circumcenter for a spherical triangle.
    Returns (radius, center) where center is the unit vector of the circumcenter
    that lies on the same side as the triangle's interior."""
    # Compute normal to plane through a, b, c
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return np.pi, None
    
    n = n / n_norm
    
    # The circumcenter is either n or -n. We pick the one that gives a smaller radius
    # but we need to ensure it's the correct one (inside the triangle's cap).
    # For points on the unit sphere forming a convex hull facet, the outward normal
    # should be the one pointing away from the origin through the facet center.
    
    # Compute the centroid of the triangle (in 3D)
    centroid = (a + b + c) / 3.0
    
    # The correct circumcenter direction is the one that has positive dot product
    # with the centroid (i.e., on the same side as the triangle)
    if np.dot(n, centroid) < 0:
        n = -n
    
    # Compute radius: arccos(dot(circumcenter, a))
    dot_val = np.dot(n, a)
    dot_val = np.clip(dot_val, -1.0, 1.0)
    radius = np.arccos(dot_val)
    
    return radius, n


def _move_point_geodesic(p, direction, step):
    """Move point p along geodesic in direction 'direction' by 'step' radians."""
    new_p = p + step * direction
    norm = np.linalg.norm(new_p)
    if norm < 1e-12:
        return p
    return new_p / norm


def _tangent_direction(p, target):
    """Get the unit tangent direction at p pointing toward target."""
    proj = target - np.dot(target, p) * p
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        return np.zeros(3)
    return proj / norm


def _covering_radius(points):
    """Compute the covering radius by finding the maximum circumradius of all Delaunay triangles."""
    try:
        tri = Delaunay(points)
    except Exception:
        return np.pi
    
    max_r = 0.0
    for simplex in tri.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        r, center = _spherical_circumradius(a, b, c)
        if center is not None:
            max_r = max(max_r, r)
    
    return max_r


def _local_refinement(points, max_iters=200):
    """Perform local refinement by moving vertices of the worst facet."""
    points = points.copy()
    
    for iteration in range(max_iters):
        # Find the worst facet
        try:
            tri = Delaunay(points)
        except Exception:
            break
        
        worst_r = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in tri.simplices:
            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]
            r, center = _spherical_circumradius(a, b, c)
            if center is not None and r > worst_r:
                worst_r = r
                worst_simplex = simplex
                worst_center = center
        
        if worst_simplex is None:
            break
        
        # Move the three vertices of the worst facet toward the circumcenter
        a = points[worst_simplex[0]]
        b = points[worst_simplex[1]]
        c = points[worst_simplex[2]]
        
        # Compute tangent directions toward the circumcenter
        dir_a = _tangent_direction(a, worst_center)
        dir_b = _tangent_direction(b, worst_center)
        dir_c = _tangent_direction(c, worst_center)
        
        # Try multiple step sizes to find the best one
        best_step = 0.0
        best_r = worst_r
        
        for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            new_a = _move_point_geodesic(a, dir_a, step)
            new_b = _move_point_geodesic(b, dir_b, step)
            new_c = _move_point_geodesic(c, dir_c, step)
            
            new_points = points.copy()
            new_points[worst_simplex[0]] = new_a
            new_points[worst_simplex[1]] = new_b
            new_points[worst_simplex[2]] = new_c
            
            new_r = _covering_radius(new_points)
            
            if new_r < best_r - 1e-10:
                best_r = new_r
                best_step = step
        
        if best_step == 0.0:
            break
        
        # Apply the best step
        a = points[worst_simplex[0]]
        b = points[worst_simplex[1]]
        c = points[worst_simplex[2]]
        
        dir_a = _tangent_direction(a, worst_center)
        dir_b = _tangent_direction(b, worst_center)
        dir_c = _tangent_direction(c, worst_center)
        
        points[worst_simplex[0]] = _move_point_geodesic(a, dir_a, best_step)
        points[worst_simplex[1]] = _move_point_geodesic(b, dir_b, best_step)
        points[worst_simplex[2]] = _move_point_geodesic(c, dir_c, best_step)
    
    return points


def construct():
    """Construct 50 points on the unit sphere with minimal covering radius."""
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Perform local refinement with more iterations
    points = _local_refinement(points, max_iters=200)
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
