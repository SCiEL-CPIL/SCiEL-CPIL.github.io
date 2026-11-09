"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull facets."""
    points = np.array(points)
    points = np.array([_normalize(p) for p in points])
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: compute max distance to nearest point on a grid
        # This is expensive but should not happen
        return np.pi  # worst case
    
    max_r = 0.0
    for simplex in hull.simplices:
        # Get the vertices of this facet
        verts = points[simplex]
        
        # Compute the circumcenter of this spherical facet
        # For a spherical triangle, the circumcenter is the point equidistant from all 3 vertices
        # We can find it by solving for the point on the sphere that maximizes the minimum angular distance
        # to the three vertices, but actually we want the circumradius which is the max angular distance
        # from the circumcenter to any vertex.
        
        # A simpler approach: the covering radius for a facet is determined by the circumcenter.
        # The circumcenter of a spherical polygon is the point on the sphere that is equidistant 
        # from all vertices. For a triangle, we can compute it.
        
        # Actually, let's use a different approach: for each facet, find the point on the sphere
        # that is farthest from all N points but closest to this facet.
        # This is complex. Instead, let's use the fact that the covering radius is the max over
        # all points on the sphere of the min distance to the N points.
        # The worst point is at the circumcenter of some facet of the convex hull.
        
        # For a spherical triangle with vertices a, b, c, the circumcenter can be found
        # by finding the point equidistant from a, b, c on the sphere.
        
        # Let's use a numerical approach: optimize to find the point on the sphere that
        # maximizes the minimum distance to the three vertices.
        
        def _neg_min_dist(x):
            x = _normalize(x)
            d = np.arccos(np.clip(np.dot(x, verts), -1.0, 1.0))
            return -np.min(d)
        
        # Start from the normalized average
        x0 = _normalize(np.sum(verts, axis=0))
        if np.linalg.norm(x0) < 1e-10:
            x0 = _normalize(verts[0])
        
        res = minimize(_neg_min_dist, x0, method='L-BFGS-B', 
                       bounds=[(-1.0, 1.0)] * 3,
                       options={'maxiter': 50, 'ftol': 1e-10})
        
        center = _normalize(res.x)
        # The circumradius is the distance from center to the farthest vertex
        d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
        max_r = max(max_r, np.max(d))
    
    return max_r


def _covering_radius_fast(points):
    """Faster covering radius estimate using a grid + hull refinement."""
    points = np.array(points)
    points = np.array([_normalize(p) for p in points])
    
    # First, get a quick estimate from a grid
    # But this is expensive. Let's just use the hull approach but with better optimization.
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_r = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        
        # For a spherical triangle, compute circumcenter analytically
        # The circumcenter of a spherical triangle can be found using spherical geometry.
        # For vertices a, b, c on the unit sphere, the circumcenter x satisfies:
        # x·a = x·b = x·c = cos(r) where r is the circumradius.
        
        # This means x is in the direction of a linear combination that makes the dot products equal.
        # We can solve: minimize ||x|| subject to x·a = x·b = x·c = 1 (then normalize)
        # Actually, we want x such that x·a = x·b = x·c, and ||x||=1.
        
        # Let's set up: find x such that x·a = x·b and x·a = x·c
        # x·(a-b) = 0 and x·(a-c) = 0
        # So x is perpendicular to (a-b) and (a-c), i.e., x is parallel to (a-b) × (a-c)
        
        v1 = verts[0] - verts[1]
        v2 = verts[0] - verts[2]
        cross = np.cross(v1, v2)
        
        if np.linalg.norm(cross) < 1e-10:
            # Degenerate case, use the normalized average
            center = _normalize(np.sum(verts, axis=0))
        else:
            center = _normalize(cross)
            # Check which side: we want the center to be on the same side as the facet normal
            # The facet normal should point outward
            # Let's check if center is "outside" by verifying it's farther from the origin
            # than the plane through the vertices... actually on the unit sphere, we just need
            # to make sure we pick the correct hemisphere.
            # The circumcenter should be such that the angular distance to each vertex is the same
            # and less than pi/2 typically for well-behaved facets.
            
            # Verify: compute distances
            d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
            # If the distances are not all close, we might have the wrong center
            # (the antipodal point)
            if np.max(d) - np.min(d) > 0.1:
                center = -center
                d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
        
        d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
        facet_r = np.max(d)
        max_r = max(max_r, facet_r)
    
    return max_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_points = np.array([_normalize(p) for p in best_points])
    
    # Compute initial covering radius
    best_cr = _covering_radius_fast(best_points)
    
    # Local optimization: iteratively move the point that is the "worst" 
    # (part of the largest facet) toward the circumcenter of that facet
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Compute hull and find worst facet
        try:
            hull = ConvexHull(best_points)
        except Exception:
            break
        
        worst_r = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in hull.simplices:
            verts = best_points[simplex]
            
            v1 = verts[0] - verts[1]
            v2 = verts[0] - verts[2]
            cross = np.cross(v1, v2)
            
            if np.linalg.norm(cross) < 1e-10:
                center = _normalize(np.sum(verts, axis=0))
            else:
                center = _normalize(cross)
                d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
                if np.max(d) - np.min(d) > 0.1:
                    center = -center
                    d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
            
            d = np.arccos(np.clip(np.dot(center, verts), -1.0, 1.0))
            facet_r = np.max(d)
            
            if facet_r > worst_r:
                worst_r = facet_r
                worst_simplex = simplex
                worst_center = center
        
        if worst_simplex is None:
            break
        
        # Move the vertices of the worst facet slightly toward the circumcenter
        # This is a Riemannian gradient step
        step_size = 0.05
        for idx in worst_simplex:
            p = best_points[idx]
            # Spherical interpolation toward center
            # Simple approach: move in the direction of center - p, projected to tangent space
            tangent_dir = worst_center - np.dot(p, worst_center) * p
            tangent_dir = _normalize(tangent_dir)
            
            # Step
            new_p = p * np.cos(step_size) + tangent_dir * np.sin(step_size)
            new_p = _normalize(new_p)
            best_points[idx] = new_p
        
        # Recompute covering radius
        current_cr = _covering_radius_fast(best_points)
        
        if current_cr < best_cr:
            best_cr = current_cr
        else:
            # Revert if it got worse
            # We need to keep track of the previous state
            # For simplicity, let's not revert but use smaller steps
            step_size *= 0.5
    
    return best_points
# EVOLVE-BLOCK-END
