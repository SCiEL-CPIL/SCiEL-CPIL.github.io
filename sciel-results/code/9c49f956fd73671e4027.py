"""Spherical Riemannian gradient flow for minimizing covering radius of 50 points."""
import numpy as np
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _spherical_circumcenter(tri):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    Returns the point on the sphere equidistant (angularly) from all three vertices.
    """
    a, b, c = tri[0], tri[1], tri[2]
    
    # The spherical circumcenter lies along the cross product of (b-a) and (c-a) in a sense,
    # but more precisely: it's the point p such that p·a = p·b = p·c.
    # This means p is orthogonal to (a-b) and (a-c) in the linear sense? No.
    # Actually, p·a = p·b means p·(a-b) = 0. So p is in the intersection of two planes.
    # The intersection of two planes through the origin is a line. The two points on the sphere
    # on that line are the circumcenter candidates.
    
    # Direction: cross product of (a-b) and (a-c) gives a vector orthogonal to both,
    # which is the direction of the line containing the circumcenter.
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    if norm < 1e-12:
        # Degenerate: points are collinear on the sphere (great circle)
        # Return midpoint of the longest edge
        d1 = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d2 = np.arccos(np.clip(np.dot(a, c), -1, 1))
        d3 = np.arccos(np.clip(np.dot(b, c), -1, 1))
        if d1 >= d2 and d1 >= d3:
            mid = a + c
        elif d2 >= d1 and d2 >= d3:
            mid = a + b
        else:
            mid = b + c
        mid_norm = np.linalg.norm(mid)
        if mid_norm < 1e-12:
            return a.copy()
        return mid / mid_norm
    
    p = cross / norm
    # Choose the sign that gives the circumcenter on the "interior" side
    # The circumcenter should be closer to the centroid of the triangle in some sense
    # Actually, we want the point that maximizes the minimum distance to the three vertices
    # among the two candidates p and -p.
    # But for a facet of the convex hull, the circumcenter is the one on the outward side.
    # For our purposes, we'll check both and pick the one that gives larger circumradius
    # (since we're trying to minimize the max circumradius, the worst case is the larger one).
    
    # Actually, let's think about this differently. The circumradius of a spherical triangle
    # is the angular distance from the circumcenter to any vertex. There are two circumcenters
    # (antipodal). The relevant one for the convex hull facet is the one on the side away
    # from the origin... no, both are on the sphere.
    
    # Let's compute the circumradius for both candidates and return the one with the larger radius
    # since that's the worst case for our covering.
    
    d1_p = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2_p = np.arccos(np.clip(np.dot(p, b), -1, 1))
    d3_p = np.arccos(np.clip(np.dot(p, c), -1, 1))
    
    d1_np = np.arccos(np.clip(np.dot(-p, a), -1, 1))
    d2_np = np.arccos(np.clip(np.dot(-p, b), -1, 1))
    d3_np = np.arccos(np.clip(np.dot(-p, c), -1, 1))
    
    # The circumradius is the max of the three distances (they should be equal for exact circumcenter)
    # Due to numerical error, they may differ slightly. Take the max.
    r_p = max(d1_p, d2_p, d3_p)
    r_np = max(d1_np, d2_np, d3_np)
    
    if r_p > r_np:
        return p
    else:
        return -p


def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull and checking each facet's circumradius.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # Fallback: worst case
    
    max_radius = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]
        cc = _spherical_circumcenter(tri)
        # Check that the circumcenter is not covered by any other point
        # (i.e., the cap is empty)
        # For now, we just compute the circumradius
        dists = np.arccos(np.clip(np.dot(cc, tri), -1, 1))
        radius = np.max(dists)
        max_radius = max(max_radius, radius)
    
    return max_radius


def _gradient_step(pts, step_size):
    """
    Perform a gradient step to reduce the maximum circumradius.
    Find the worst facet, compute its circumcenter, and move the vertices
    of that facet toward the circumcenter on the sphere.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts.copy()
    
    # Find the facet with the largest circumradius
    worst_idx = -1
    worst_radius = 0.0
    worst_tri = None
    worst_cc = None
    
    for i, simplex in enumerate(hull.simplices):
        tri = pts[simplex]
        cc = _spherical_circumcenter(tri)
        dists = np.arccos(np.clip(np.dot(cc, tri), -1, 1))
        radius = np.max(dists)
        if radius > worst_radius:
            worst_radius = radius
            worst_idx = i
            worst_tri = simplex
            worst_cc = cc
    
    if worst_idx == -1:
        return pts.copy()
    
    # Move the vertices of the worst facet toward the circumcenter
    new_pts = pts.copy()
    for idx in worst_tri:
        p = new_pts[idx]
        cc = worst_cc
        
        # Spherical gradient: move p toward cc
        # The direction on the sphere from p to cc is given by the projection of (cc - p)
        # onto the tangent space at p, which is (cc - (p·cc)p)
        dot_pc = np.dot(p, cc)
        tangent_dir = cc - dot_pc * p
        tangent_norm = np.linalg.norm(tangent_dir)
        
        if tangent_norm > 1e-12:
            tangent_dir = tangent_dir / tangent_norm
            # Move a small step toward the circumcenter
            new_p = p + step_size * tangent_dir
            # Normalize back to the sphere
            new_pts[idx] = _normalize(new_p.reshape(1, -1))[0]
    
    return new_pts


def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Optimize using gradient flow
    current_radius = _compute_covering_radius(pts)
    
    # Adaptive step size
    step_size = 0.01
    best_pts = pts.copy()
    best_radius = current_radius
    
    for iteration in range(5000):
        new_pts = _gradient_step(pts, step_size)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            # Decrease step size slightly as we improve
            step_size = max(0.001, step_size * 0.999)
        else:
            # Reduce step size if we didn't improve
            step_size = max(0.0005, step_size * 0.9)
        
        # Check convergence
        if best_radius < 0.31937:  # ~18.3 degrees in radians
            break
    
    return best_pts

# EVOLVE-BLOCK-END
