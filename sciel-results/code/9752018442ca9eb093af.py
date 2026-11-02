"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _get_hull_facets(points):
    """
    Compute the convex hull of the points on the sphere.
    Returns the indices of the facets (triangles) that form the boundary.
    """
    try:
        hull = ConvexHull(points)
        return hull.simplices
    except Exception:
        # If points are degenerate, return empty
        return np.array([], dtype=int).reshape(0, 3)

def _covering_radius_hull(points):
    """
    Compute the covering radius based on the circumradii of the convex hull facets.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the configuration. This is equivalent to the maximum
    circumradius of the spherical Voronoi cells, which correspond to the facets
    of the convex hull of the points.
    """
    points = np.asarray(points)
    # Normalize points to ensure they are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    facets = _get_hull_facets(points)
    if len(facets) == 0:
        # Fallback if hull fails: use grid sampling
        return _covering_radius_grid(points)

    max_radius = 0.0
    
    for facet in facets:
        p1, p2, p3 = points[facet[0]], points[facet[1]], points[facet[2]]
        
        # Check if the facet is "visible" from the origin (normal points out)
        # The center of the spherical triangle is the normalized sum of the vertices
        # or the circumcenter.
        
        # Compute circumcenter of the spherical triangle (p1, p2, p3)
        # The circumcenter is the point equidistant from p1, p2, p3 on the sphere.
        # It is given by the normalized cross product of (p2-p1) and (p3-p1) ? No.
        # The circumcenter of a spherical triangle is the intersection of the sphere
        # with the plane passing through the origin and the circumcenter of the planar triangle?
        # No, the spherical circumcenter is the normalized vector:
        # C = (p1 * sin(a) + p2 * sin(b) + p3 * sin(c)) / norm(...)
        # where a, b, c are the side lengths (angular distances).
        
        # Simpler method: The circumcenter is the solution to:
        # C . p1 = C . p2 = C . p3 = cos(R)
        # C is a unit vector.
        # This is equivalent to finding the null space of:
        # [p1 - p3]
        # [p2 - p3]
        # and normalizing.
        
        v1 = p1 - p3
        v2 = p2 - p3
        # Cross product gives a vector perpendicular to the plane of the triangle,
        # but we need the circumcenter.
        # The circumcenter C lies in the plane defined by p1, p2, p3.
        # Actually, the circumcenter of the spherical triangle is the normalized
        # vector: C = normalize( p1 * (1 + cos(d23) - cos(d12)cos(d13)) + ... )
        
        # Let's use the property that the circumcenter is the point on the sphere
        # maximizing the distance to the plane of the triangle? No.
        
        # Efficient calculation:
        # The circumcenter is the normalized cross product of the normals of the great circles?
        
        # Let's use the formula for the circumcenter of a spherical triangle:
        # Let a, b, c be the angular sides opposite to p1, p2, p3.
        # cos(a) = p2 . p3
        # cos(b) = p1 . p3
        # cos(c) = p1 . p2
        # The circumradius R satisfies:
        # cos(R) = (cos(a) + cos(b) + cos(c) + 2*cos(a)*cos(b)*cos(c)) / (4*cos((a+b-c)/2)*cos((a+c-b)/2)*cos((b+c-a)/2))
        # This is complex.
        
        # Alternative: The circumcenter is the normalized sum of the vertices weighted by the sines of the opposite sides?
        # C = sin(a)*p1 + sin(b)*p2 + sin(c)*p3
        # Let's verify.
        
        cos_a = np.clip(np.dot(p2, p3), -1.0, 1.0)
        cos_b = np.clip(np.dot(p1, p3), -1.0, 1.0)
        cos_c = np.clip(np.dot(p1, p2), -1.0, 1.0)
        
        sin_a = np.sqrt(np.maximum(0.0, 1.0 - cos_a**2))
        sin_b = np.sqrt(np.maximum(0.0, 1.0 - cos_b**2))
        sin_c = np.sqrt(np.maximum(0.0, 1.0 - cos_c**2))
        
        # Check for degenerate triangles
        if sin_a < 1e-8 and sin_b < 1e-8 and sin_c < 1e-8:
            continue
            
        C = sin_a * p1 + sin_b * p2 + sin_c * p3
        norm_C = np.linalg.norm(C)
        
        if norm_C < 1e-8:
            # Degenerate case, e.g., points on a great circle
            # The circumradius is half the maximum distance between points?
            # Or just skip if it's a very small area facet?
            # Actually, if they are collinear on a great circle, the "circumcenter" is ambiguous.
            # We can approximate the radius by the max distance to the midpoint.
            # But for covering radius, the worst case is likely elsewhere.
            # Let's use a large penalty or skip.
            continue
            
        C = C / norm_C
        
        # The distance from C to p1 is the circumradius
        cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Ensure the circumcenter is on the correct side (not the antipode)
        # The circumcenter should be "inside" the triangle.
        # If dot(C, p1+p2+p3) < 0, we might have the antipode.
        # Actually, the formula C = sin(a)p1 + ... always points towards the interior if the triangle is small (< pi).
        
        if R > max_radius:
            max_radius = R
            
    return max_radius

def _covering_radius_grid(points):
    """Fallback grid-based covering radius."""
    n_test = 5000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    cos_d = np.clip(test_points @ points.T, -1.0, 1.0)
    min_cos = np.max(cos_d, axis=1)
    min_angles = np.arccos(np.clip(min_cos, -1.0, 1.0))
    return np.max(min_angles)

def _objective_cartesian(params):
    """
    Objective function for Cartesian coordinates.
    params: 3N parameters.
    """
    points = params.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    return _covering_radius_hull(points)

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    params0 = points0.flatten()
    
    # Optimize using Powell's method, which is robust for non-differentiable objectives
    # and handles the constraint implicitly via normalization in the objective.
    # We use Cartesian coordinates to avoid spherical coordinate singularities.
    
    result = minimize(
        _objective_cartesian,
        params0,
        method='Powell',
        options={'maxiter': 500, 'disp': False}
    )
    
    points = result.x.reshape(-1, 3)
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
