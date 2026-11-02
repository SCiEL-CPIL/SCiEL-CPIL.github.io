"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    # Get the convex hull facets
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0  # Invalid configuration
    
    # For each facet, compute the circumradius
    # The covering radius is the max over all points on the sphere of distance to nearest point
    # This equals the max circumradius of the Voronoi cells (which are related to hull facets)
    
    # More directly: the covering radius is the maximum, over all directions y, of the 
    # angular distance to the nearest point. This is achieved at the circumcenters of 
    # the spherical Voronoi cells, which correspond to the dual of the convex hull.
    
    # For a point y on the sphere, the nearest point is determined by which Voronoi cell it's in.
    # The worst case is at the vertices of the Voronoi diagram (circumcenters of hull facets).
    
    # Each hull facet corresponds to a set of points. The circumcenter of that facet (projected to sphere)
    # is a candidate for the covering radius.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # Compute circumcenter of the facet
        # For a triangle, the circumcenter in 3D
        a, b, c = facet_points
        
        # Circumcenter of triangle abc
        # Using the formula: circumcenter = (a|b|^2*(c-a) + b|c|^2*(a-b) + c|a|^2*(b-c)) / (2 * cross(a-b, c-b) · (a-b))... 
        # Actually, let's use a more robust method.
        
        # The circumcenter of points a, b, c lies on the plane through them.
        # Find point p such that |p-a|^2 = |p-b|^2 = |p-c|^2
        # This gives 2 linear equations in 3 unknowns, plus we want p on the sphere if possible,
        # but for the angular distance we just need the direction.
        
        # Actually, the farthest point from the set {a,b,c} within their Voronoi cell is the 
        # circumcenter of the triangle abc (if it's inside the cell) or a vertex.
        
        # Let me compute the circumcenter of the triangle abc in 3D.
        # The circumcenter satisfies:
        # |p - a|^2 = |p - b|^2 => 2 p·(b-a) = |b|^2 - |a|^2
        # |p - a|^2 = |p - c|^2 => 2 p·(c-a) = |c|^2 - |a|^2
        
        # Since a, b, c are on the unit sphere, |a|^2 = |b|^2 = |c|^2 = 1
        # So: 2 p·(b-a) = 0 and 2 p·(c-a) = 0
        # Thus p is orthogonal to both (b-a) and (c-a), i.e., p is parallel to (b-a) × (c-a)
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        p_dir = n / n_norm
        
        # The circumcenter is at p_dir or -p_dir; we want the one that maximizes distance
        # from the facet points, but actually we want the point on the sphere that is 
        # equidistant to a, b, c.
        
        # The angular distance from p_dir to a is arccos(p_dir · a)
        # Check both p_dir and -p_dir
        d1 = np.arccos(np.clip(np.dot(p_dir, a), -1, 1))
        d2 = np.arccos(np.clip(-np.dot(p_dir, a), -1, 1))
        
        # The circumcenter should be equidistant to a, b, c. Let's verify.
        # The actual circumcenter on the sphere is the point equidistant to a, b, c.
        # It's along the direction of the normal to the plane through a,b,c.
        
        # The angular distance from this circumcenter to a (and b, c) is the circumradius of the facet.
        # We want the one that is "outside" the triangle in the sense of being the center of the cap.
        
        # Actually, for the covering radius, we need the point on the sphere that is farthest 
        # from all 50 points. This will be at a Voronoi vertex, which is the circumcenter of 
        # some hull facet.
        
        # The circumcenter of the hull facet (in the dual sense) is the point on the sphere 
        # equidistant to the facet's points. There are two such points (antipodal). 
        # The relevant one is the one that is "opposite" to the hull center.
        
        # Let's compute the angular distance from p_dir to a:
        # If p_dir is the circumcenter, then the angular distance is arccos(p_dir · a)
        
        # But we need to make sure this is the correct circumcenter. Let's check if it's 
        # equidistant to all three.
        da = np.arccos(np.clip(np.dot(p_dir, a), -1, 1))
        db = np.arccos(np.clip(np.dot(p_dir, b), -1, 1))
        dc = np.arccos(np.clip(np.dot(p_dir, c), -1, 1))
        
        if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
            # This is a valid circumcenter
            # The covering radius candidate is da (or the max of da, db, dc, but they should be equal)
            max_radius = max(max_radius, da)
        else:
            # Try the other direction
            da = np.arccos(np.clip(-np.dot(p_dir, a), -1, 1))
            db = np.arccos(np.clip(-np.dot(p_dir, b), -1, 1))
            dc = np.arccos(np.clip(-np.dot(p_dir, c), -1, 1))
            if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
                max_radius = max(max_radius, da)
    
    # Also check the points themselves? No, the covering radius is determined by the 
    # circumcenters of the Voronoi cells, not the points themselves.
    
    return np.degrees(max_radius)


def _objective(params):
    """Objective function: covering radius in radians."""
    # Convert parameters to points
    # Use spherical coordinates: theta in [0, pi], phi in [0, 2*pi)
    theta = params[0::2]
    phi = params[1::2]
    
    # Map to sphere
    points = np.zeros((N, 3))
    points[:, 0] = np.sin(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.sin(phi)
    points[:, 2] = np.cos(theta)
    
    # Compute covering radius
    try:
        cr = _covering_radius(points)
    except Exception:
        return 180.0
    
    return np.radians(cr)


def _covering_radius_fast(points):
    """Faster covering radius computation using vectorized operations."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_points = points[simplex]
        a, b, c = facet_points
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        p_dir = n / n_norm
        
        da = np.arccos(np.clip(np.dot(p_dir, a), -1, 1))
        db = np.arccos(np.clip(np.dot(p_dir, b), -1, 1))
        dc = np.arccos(np.clip(np.dot(p_dir, c), -1, 1))
        
        if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
            max_radius = max(max_radius, da)
        else:
            da = np.arccos(np.clip(-np.dot(p_dir, a), -1, 1))
            db = np.arccos(np.clip(-np.dot(p_dir, b), -1, 1))
            dc = np.arccos(np.clip(-np.dot(p_dir, c), -1, 1))
            if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
                max_radius = max(max_radius, da)
    
    return np.degrees(max_radius)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates for optimization
    theta = np.arccos(np.clip(points[:, 2], -1, 1))
    phi_coords = np.arctan2(points[:, 1], points[:, 0])
    
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi_coords
    
    best_params = params.copy()
    best_score = _covering_radius_fast(points)
    
    # Try L-BFGS optimization
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline - 20:
            return points
        
        # Evaluate result
        theta_res = result.x[0::2]
        phi_res = result.x[1::2]
        points_res = np.zeros((N, 3))
        points_res[:, 0] = np.sin(theta_res) * np.cos(phi_res)
        points_res[:, 1] = np.sin(theta_res) * np.sin(phi_res)
        points_res[:, 2] = np.cos(theta_res)
        
        score_res = _covering_radius_fast(points_res)
        if score_res < best_score:
            best_score = score_res
            best_params = result.x.copy()
            points = points_res.copy()
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 30:
            break
        
        # Random initial configuration
        params_rand = np.random.uniform(0, np.pi, N)
        phi_rand = np.random.uniform(0, 2 * np.pi, N)
        params_full = np.zeros(2 * N)
        params_full[0::2] = params_rand
        params_full[1::2] = phi_rand
        
        try:
            result = minimize(
                _objective,
                params_full,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            theta_res = result.x[0::2]
            phi_res = result.x[1::2]
            points_res = np.zeros((N, 3))
            points_res[:, 0] = np.sin(theta_res) * np.cos(phi_res)
            points_res[:, 1] = np.sin(theta_res) * np.sin(phi_res)
            points_res[:, 2] = np.cos(theta_res)
            
            score_res = _covering_radius_fast(points_res)
            if score_res < best_score:
                best_score = score_res
                best_params = result.x.copy()
                points = points_res.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 15:
        try:
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            theta_res = result.x[0::2]
            phi_res = result.x[1::2]
            points_res = np.zeros((N, 3))
            points_res[:, 0] = np.sin(theta_res) * np.cos(phi_res)
            points_res[:, 1] = np.sin(theta_res) * np.sin(phi_res)
            points_res[:, 2] = np.cos(theta_res)
            
            score_res = _covering_radius_fast(points_res)
            if score_res < best_score:
                best_score = score_res
                points = points_res.copy()
        except Exception:
            pass
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
