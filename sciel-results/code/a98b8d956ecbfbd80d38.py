"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _cover_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest chosen point. This equals the largest circumradius of the facets
    of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the vertices of this facet (triangle)
            v = points[simplex]
            # Compute the circumradius of the spherical triangle
            # For a spherical triangle with vertices a, b, c on the unit sphere,
            # the circumradius (angular) can be found by finding the point equidistant
            # from all three vertices.
            
            a, b, c = v[0], v[1], v[2]
            
            # The center of the circumscribed circle on the sphere is in the direction
            # of a cross-product-based normal to the plane through a, b, c, but projected
            # to the sphere. Actually, for points on the unit sphere, the circumcenter
            # of the spherical triangle is along the direction of (a x b) x (b x c)
            # normalized... let me think more carefully.
            
            # A simpler approach: the circumradius of the spherical triangle is the
            # angular distance from the circumcenter to any vertex. The circumcenter
            # lies along the direction of the cross product of (b-a) and (c-a) normalized,
            # but that gives the plane normal. For the unit sphere, the circumcenter of
            # a spherical triangle with vertices a, b, c is in the direction of
            # (a x b) x (b x c) ... no.
            
            # Let me use a different approach: solve for the point x on the unit sphere
            # such that |x-a| = |x-b| = |x-c| (Euclidean distances, which correspond
            # to angular distances via cos).
            
            # |x-a|^2 = |x-b|^2 => 2 x·(b-a) = |b|^2 - |a|^2 = 0 (since unit vectors)
            # So x·(b-a) = 0 and x·(c-a) = 0, meaning x is perpendicular to the plane
            # spanned by (b-a) and (c-a), i.e., x is along the normal to the plane through
            # a, b, c.
            
            # The normal to the plane through a, b, c is n = (b-a) x (c-a)
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                # Degenerate triangle
                continue
            n = n / n_norm
            
            # The circumcenter is either n or -n. We need the one on the same side
            # as the "inside" of the hull. For the covering radius, we want the cap
            # that is empty, so the circumcenter should be on the side opposite to
            # the origin for the facet's cap.
            
            # Actually, for the convex hull of points on the unit sphere, each facet
            # corresponds to a cap. The circumcenter of the spherical triangle is the
            # point on the sphere farthest from all three vertices within that cap.
            # It's along the direction of the outward normal of the facet.
            
            # The outward normal of the facet (pointing away from the origin) is
            # in the direction of the centroid of the triangle, roughly.
            centroid = (a + b + c) / 3.0
            # Determine sign
            if np.dot(n, centroid) < 0:
                n = -n
            
            # The angular distance from the circumcenter to a vertex
            # cos(angle) = n · a
            cos_angle = np.dot(n, a)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            
            # Convert to degrees
            angle_deg = np.degrees(angle)
            if angle_deg > max_radius:
                max_radius = angle_deg
        
        return max_radius
    except Exception:
        return 180.0  # Fallback: worst case


def _cover_radius_fast(points):
    """Faster covering radius computation using the property that the covering radius
    equals the maximum over facets of the convex hull of the angular distance from
    the facet's circumcenter to its vertices."""
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            v = points[simplex]
            a, b, c = v[0], v[1], v[2]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            
            cos_angle = np.dot(n, a)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            angle_deg = np.degrees(angle)
            if angle_deg > max_radius:
                max_radius = angle_deg
        
        return max_radius
    except Exception:
        return 180.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _cover_radius(points)
    
    # Try to improve using local optimization
    # We'll use a simple iterative approach: for each point, try small perturbations
    # and keep the one that reduces the covering radius.
    
    # First, let's try a few random restarts with local refinement
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Use the Fibonacci spiral
            current = best_points.copy()
        else:
            # Random initialization with some structure
            # Use a perturbed Fibonacci spiral
            noise_scale = 0.3 * attempt
            current = best_points + rng.normal(0, noise_scale, size=(N, 3))
            current = current / np.linalg.norm(current, axis=1, keepdims=True)
        
        # Local refinement: iterative nearest-point improvement
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                
                # Try moving point idx slightly in various directions
                p = current[idx]
                # Generate random tangent directions
                for trial in range(10):
                    # Random tangent vector
                    rand_vec = rng.normal(size=3)
                    tangent = rand_vec - np.dot(rand_vec, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm < 1e-12:
                        continue
                    tangent = tangent / t_norm
                    
                    # Try a few step sizes
                    for step in [0.05, 0.02, 0.01]:
                        # Move along tangent
                        angle = step
                        new_p = (np.cos(angle) * p + np.sin(angle) * tangent)
                        new_p = new_p / np.linalg.norm(new_p)
                        
                        candidate = current.copy()
                        candidate[idx] = new_p
                        
                        new_radius = _cover_radius(candidate)
                        if new_radius < best_radius - 1e-10:
                            current = candidate
                            best_radius = new_radius
                            best_points = current.copy()
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
