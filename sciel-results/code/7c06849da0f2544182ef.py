"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _compute_covering_radius(points):
    """
    Compute the covering radius using the convex hull.
    The covering radius is the maximum circumradius of the triangular facets of the convex hull.
    For each facet, the circumcenter projected to the sphere gives the farthest point from the facet's vertices.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices  # shape (M, 3)
        
        max_radius = 0.0
        for tri in facets:
            a = points[tri[0]]
            b = points[tri[1]]
            c = points[tri[2]]
            
            # Compute circumcenter of triangle abc on the unit sphere
            # The circumcenter is equidistant from a, b, c
            # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2, |x| = 1
            # This gives: x dot (b - a) = (|b|^2 - |a|^2) / 2 = 0 (since |a|=|b|=1)
            # Similarly: x dot (c - a) = 0
            # So x is perpendicular to (b-a) and (c-a), i.e., x is parallel to (b-a) x (c-a)
            
            v1 = b - a
            v2 = c - a
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-10:
                continue
            
            # The circumcenter direction is along the cross product
            # But we need the one that's on the "outside" (away from origin for convex hull)
            # Actually for points on unit sphere, the circumcenter of the spherical triangle
            # is in the direction of the normalized cross product, but we need to pick the right sign
            
            # The circumcenter x satisfies x dot a = x dot b = x dot c
            # Let x = t * cross / |cross|
            # x dot a = t * (cross dot a) / |cross|
            # We need to find t such that |x| = 1, so t = |cross|
            # Thus x = cross / |cross| or x = -cross / |cross|
            
            # Determine which sign gives the correct circumcenter
            # The circumcenter should be on the same side as the origin for the "inside"
            # Actually, for the covering radius, we want the point on the sphere farthest from the triangle
            # which is the circumcenter on the opposite side from the triangle's interior
            
            # Let's compute both and pick the one with larger distance to the vertices
            x1 = cross / norm_cross
            x2 = -cross / norm_cross
            
            # The circumradius is the angular distance from x to any of a, b, c
            # cos(angle) = x dot a
            cos1 = np.abs(np.dot(x1, a))
            cos2 = np.abs(np.dot(x2, a))
            
            # The circumradius is arccos(min_cos) where min_cos is the minimum cos over a,b,c
            # Actually, the circumradius is the angle from the circumcenter to the nearest vertex
            # So we want the circumcenter that maximizes this angle (i.e., minimizes the cos)
            
            # For x1: cos to a, b, c
            cos_a1 = np.dot(x1, a)
            cos_b1 = np.dot(x1, b)
            cos_c1 = np.dot(x1, c)
            min_cos1 = min(cos_a1, cos_b1, cos_c1)
            
            cos_a2 = np.dot(x2, a)
            cos_b2 = np.dot(x2, b)
            cos_c2 = np.dot(x2, c)
            min_cos2 = min(cos_a2, cos_b2, cos_c2)
            
            # The covering radius for this facet is max(arccos(min_cos1), arccos(min_cos2))
            # But actually, the empty cap is centered at the circumcenter on the side opposite to the hull interior
            # For a convex hull of points on the unit sphere, the "outside" is away from the origin
            # The circumcenter that's farther from the origin side is the one we want
            
            # Simpler: the covering radius is the angular distance from the circumcenter to the nearest vertex
            # We want the maximum over all facets
            radius1 = np.arccos(np.clip(min_cos1, -1, 1))
            radius2 = np.arccos(np.clip(min_cos2, -1, 1))
            
            # The relevant circumcenter is the one on the "outer" side
            # For convex hull facets, the normal points outward
            # The circumcenter on the outer side has a larger angle to the vertices
            facet_radius = max(radius1, radius2)
            
            if facet_radius > max_radius:
                max_radius = facet_radius
        
        return max_radius
    except Exception:
        # Fallback: use a grid-based estimate
        return _grid_covering_radius(points)

def _grid_covering_radius(points):
    """Fallback: estimate covering radius by sampling the sphere."""
    # Create a grid of test points
    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    
    test_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute distances from each test point to each generator
    # diff shape (M, N, 3)
    diff = test_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Minimum distance for each test point
    min_dists = np.min(dists, axis=1)
    
    # Convert to angular distance
    cos_angles = 1.0 - 0.5 * min_dists ** 2
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)
    
    return np.max(angles)

def construct():
    # Start with a good initial guess
    points = _fibonacci_sphere(N)
    
    # Compute initial covering radius
    start_time = time.time()
    time_limit = 115.0  # Leave margin for 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization: use gradient-free optimization (Nelder-Mead or similar)
    # Or use a simple perturbation-based local search
    
    np.random.seed(42)
    
    # Use a combination of global and local search
    # Phase 1: Coarse optimization with random perturbations
    # Phase 2: Fine optimization with smaller perturbations
    
    phase1_steps = 5000
    phase2_steps = 10000
    
    for step in range(phase1_steps + phase2_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Determine step size based on phase
        if step < phase1_steps:
            step_size = 0.1 * (1.0 - step / phase1_steps) + 0.01
        else:
            step_size = 0.01 * (1.0 - (step - phase1_steps) / phase2_steps) + 0.001
        
        # Random perturbation
        perturbations = np.random.randn(N, 3) * step_size
        new_points = points + perturbations
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1)
        new_points = new_points / norms[:, np.newaxis]
        
        # Check for degenerate cases (coincident points, hemisphere)
        # Check if all points are in a closed hemisphere
        # A set of points is in a closed hemisphere if there exists a vector v such that v dot p_i >= 0 for all i
        # This is equivalent to the origin being outside the convex hull
        try:
            hull = ConvexHull(new_points)
            # Check if origin is inside the convex hull
            # If all facet normals point away from origin, origin is outside
            # For a convex hull containing the origin, the origin is inside
            # Simple check: the convex hull of points on unit sphere contains origin iff points are not in a hemisphere
            # We can check by seeing if the minimum distance from origin to any facet is negative
            # Actually, scipy's ConvexHull doesn't directly tell us if origin is inside
            # Let's use a different check: if the points span R^3 and the origin is in their convex hull
            # For now, skip this check and rely on the covering radius being large if in hemisphere
        except Exception:
            continue
        
        # Compute new covering radius
        new_radius = _compute_covering_radius(new_points)
        
        # Accept if better
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
