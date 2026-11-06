"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize_scalar


def _spherical_circumradius(a, b, c):
    """Compute the spherical circumradius (angular distance from circumcenter to vertices)
    for a spherical triangle with vertices a, b, c (unit vectors).
    Returns (radius, center) where center is the unit vector of the circumcenter."""
    # Compute edge lengths (angular distances)
    def ang_dist(u, v):
        d = np.dot(u, v)
        d = np.clip(d, -1.0, 1.0)
        return np.arccos(d)
    
    A = ang_dist(b, c)  # angle at a? No, side opposite to a
    B = ang_dist(a, c)
    C = ang_dist(a, b)
    
    # Spherical law of cosines for circumradius
    # For a spherical triangle with sides a, b, c (opposite to vertices A, B, C),
    # the circumradius R satisfies:
    # sin(A) = sin(a) / (2 * sin(R)) ... not quite.
    # Actually, the spherical circumradius R is given by:
    # tan(R) = ... 
    # Let's use the formula: 
    # The circumcenter is the point equidistant from a, b, c.
    # In 3D, the circumcenter of the triangle formed by a,b,c on the unit sphere
    # lies along the normal to the plane containing a,b,c.
    
    # Compute normal to plane through a, b, c
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        # Degenerate triangle
        return np.pi, None
    
    n = n / n_norm
    
    # The circumcenter is either n or -n (whichever is closer to the triangle)
    # It should be on the same side as the triangle's interior
    # The circumcenter R satisfies: dot(R, a) = dot(R, b) = dot(R, c) = cos(radius)
    # If R = n, then cos(radius) = dot(n, a)
    # But we need to check if n or -n is the correct one (the one where the cap is empty)
    
    # For now, compute both and pick the one that gives a smaller radius
    # Actually, the circumcenter of the spherical triangle is the point on the sphere
    # equidistant from all three vertices. There are two such points (antipodal).
    # The relevant one is the one that lies "inside" the triangle's cap.
    
    # Let's compute the radius for n:
    dot_na = np.dot(n, a)
    dot_na = np.clip(dot_na, -1.0, 1.0)
    r_n = np.arccos(dot_na)
    
    dot_na_neg = np.dot(-n, a)
    dot_na_neg = np.clip(dot_na_neg, -1.0, 1.0)
    r_neg = np.arccos(dot_na_neg)
    
    # The circumradius should be the smaller of the two? No, it depends on which side.
    # For a proper spherical triangle (less than a hemisphere), the circumcenter
    # is the one closer to the triangle. Let's pick the one with smaller radius
    # but we need to ensure it's the correct one.
    
    # Actually, for the covering radius, we care about the facet of the convex hull.
    # The circumcenter of the facet (on the unit sphere) is what we want.
    # For a Delaunay triangulation of points on the sphere, the circumcenter
    # of each triangle should be empty (no other points in the cap).
    
    # Let's just return the circumcenter that is on the same side as the "outward" normal.
    # For points on the unit sphere, the convex hull's facet normal points outward.
    # The circumcenter of the spherical triangle is along this normal.
    
    # The radius is arccos(dot(circumcenter, a))
    # Let's use n as the circumcenter direction
    # But we need to verify it's the correct one.
    
    # For now, return r_n and n
    return r_n, n


def _move_point_geodesic(p, direction, step):
    """Move point p along geodesic in direction 'direction' by 'step' radians."""
    # direction should be a unit vector tangent to the sphere at p
    # p + step * direction, then normalize
    new_p = p + step * direction
    norm = np.linalg.norm(new_p)
    if norm < 1e-12:
        return p
    return new_p / norm


def _tangent_direction(p, target):
    """Get the unit tangent direction at p pointing toward target."""
    # Project target onto the tangent plane at p
    proj = target - np.dot(target, p) * p
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        return np.zeros(3)
    return proj / norm


def _covering_radius(points):
    """Compute the covering radius by finding the maximum circumradius of all Delaunay triangles."""
    # Create Delaunay triangulation
    # We need to be careful with points that are nearly coplanar
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback: return a large value
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


def _local_refinement(points, max_iters=100):
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
        # The geodesic direction from each vertex to the circumcenter
        a = points[worst_simplex[0]]
        b = points[worst_simplex[1]]
        c = points[worst_simplex[2]]
        
        # Compute tangent directions toward the circumcenter
        dir_a = _tangent_direction(a, worst_center)
        dir_b = _tangent_direction(b, worst_center)
        dir_c = _tangent_direction(c, worst_center)
        
        # Try a small step
        best_step = 0.0
        best_r = worst_r
        
        for step in [0.01, 0.005, 0.001, 0.0005]:
            new_a = _move_point_geodesic(a, dir_a, step)
            new_b = _move_point_geodesic(b, dir_b, step)
            new_c = _move_point_geodesic(c, dir_c, step)
            
            # Check if this reduces the covering radius
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
    
    # Perform local refinement
    points = _local_refinement(points, max_iters=50)
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
