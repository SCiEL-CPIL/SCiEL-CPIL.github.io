"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: kill after 150s wall-clock, return best found with safety margin
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius (max angular distance to nearest point).
    
    Uses the convex hull approach: the covering radius is the maximum circumradius
    of the spherical triangles (facets of the convex hull). For a set of points on
    the unit sphere, the covering radius equals the maximum over all facets of the
    circumradius of that facet (the smallest cap containing the facet vertices).
    
    For practical purposes with 50 points, we can compute this by:
    1. Computing the Voronoi diagram on the sphere (or equivalently, the convex hull)
    2. For each Voronoi cell, the covering radius is the max distance from the 
       generating point to the cell vertices.
    
    A simpler and robust approach: for each pair of nearby points, compute the 
    midpoint direction, and check if that direction is closer to any other point.
    The covering radius is the maximum over all such "worst-case" directions.
    
    Actually, the most robust method: the covering radius is the smallest r such
    that caps of angular radius r centered at each point cover the sphere.
    This equals the maximum, over all directions y on the sphere, of min_i angle(y, p_i).
    
    For a finite set, the worst-case y lies at a vertex of the Voronoi diagram on
    the sphere, which corresponds to a point equidistant (in angle) to at least 3
    of the generating points (or 2 if on an edge). These are the circumcenters of
    spherical triangles formed by triples of points.
    
    Practical computation: 
    - For each triple of points, compute the spherical circumcenter.
    - Check if the circumcenter is actually the nearest-neighbor Voronoi vertex
      (i.e., the three points are indeed the three nearest to that direction).
    - The covering radius is the max angle from the circumcenter to its nearest point.
    
    This is O(N^3) which for N=50 is 125000 triples - feasible.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # For each triple, compute the point equidistant to all three (spherical circumcenter)
    # and verify it's a valid Voronoi vertex.
    best_r = 0.0
    
    # Precompute all pairwise dot products
    dots = pts @ pts.T
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Compute the point on the sphere equidistant to pts[i], pts[j], pts[k]
                # This is the circumcenter of the spherical triangle.
                # In 3D: the circumcenter is along the direction of the cross product
                # of (pts[i]-pts[j]) x (pts[i]-pts[k])... 
                # Actually, the point equidistant to a, b, c on the sphere satisfies:
                # |x|^2 = 1, x.a = x.b = x.c = cos(r)
                # This means x is in the null space of [a-b; b-c] (two constraints)
                # So x is parallel to (a-b) x (b-c), normalized.
                
                a, b, c = pts[i], pts[j], pts[k]
                # Direction of circumcenter
                v1 = a - b
                v2 = b - c
                x = np.cross(v1, v2)
                nx = np.linalg.norm(x)
                if nx < 1e-10:
                    continue
                x = x / nx
                
                # Also check the opposite direction
                x_neg = -x
                
                for xc in [x, x_neg]:
                    # Distance to each of the three points
                    d_i = np.arccos(np.clip(xc @ a, -1, 1))
                    d_j = np.arccos(np.clip(xc @ b, -1, 1))
                    d_k = np.arccos(np.clip(xc @ c, -1, 1))
                    
                    # For this to be a Voronoi vertex, the three points must be the
                    # three nearest to xc. Check that no other point is closer.
                    all_dists = np.arccos(np.clip(pts @ xc, -1, 1))
                    min_dist = np.min(all_dists)
                    min_idx = np.argmin(all_dists)
                    
                    # The three points should be among the closest
                    # Actually, for a proper Voronoi vertex, exactly 3 (or more) points
                    # should be at the minimum distance. Let's check:
                    if min_dist > max(d_i, d_j, d_k) + 1e-9:
                        continue
                    # Check that i, j, k are all at the minimum distance (within tolerance)
                    if not (abs(all_dists[i] - min_dist) < 1e-6 and 
                            abs(all_dists[j] - min_dist) < 1e-6 and 
                            abs(all_dists[k] - min_dist) < 1e-6):
                        continue
                    
                    # This is a valid Voronoi vertex. The covering radius contribution
                    # is min_dist (distance to nearest point).
                    if min_dist > best_r:
                        best_r = min_dist
    
    # Also need to check: the worst point might be at a point where only 2 points
    # are equidistant (on an edge of the Voronoi diagram). But typically for generic
    # configurations, the worst point is at a vertex. For safety, also check midpoints
    # of pairs that are close to each other.
    
    # For each pair, the midpoint direction (bisector) is a candidate if it's the
    # second-closest point's direction.
    for i in range(n):
        for j in range(i+1, n):
            mid = pts[i] + pts[j]
            nm = np.linalg.norm(mid)
            if nm < 1e-10:
                continue
            mid = mid / nm
            
            all_dists = np.arccos(np.clip(pts @ mid, -1, 1))
            min_dist = np.min(all_dists)
            # Check if i and j are the two closest
            sorted_dists = np.sort(all_dists)
            if (abs(sorted_dists[0] - min_dist) < 1e-9 and 
                abs(sorted_dists[1] - min_dist) < 1e-6):
                if min_dist > best_r:
                    best_r = min_dist
    
    return best_r

def _log_sum_exp(x, axis=-1):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def _smooth_covering_radius(pts, beta=50.0):
    """Smooth surrogate for the covering radius using log-sum-exp.
    
    The covering radius is max_y min_i angle(y, p_i).
    We approximate max_y by optimizing over y, and min_i by a soft-min.
    
    Actually, for the objective to minimize, we want to minimize the covering radius.
    The covering radius = max over y of min over i of angle(y, p_i).
    
    A smooth approximation: for a given configuration, the covering radius is the
    smallest r such that all caps of radius r cover the sphere. 
    Equivalently, it's the max over Voronoi vertices of the distance to the nearest point.
    
    For optimization, we can use: for each point y (sampled or optimized), 
    f(y) = -logsumexp(-beta * angle(y, p_i) for i) / beta  approximates min_i angle(y, p_i)
    Then the covering radius approx = max_y f(y).
    
    But we can't easily maximize over y in the objective. Instead, we can use the
    dual formulation or just optimize the points to minimize the max over a set
    of test directions.
    
    Alternative: use the fact that the covering radius equals the max circumradius
    of spherical triangles. We can smooth this by considering all triples and using
    a soft-max over their circumradii.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute circumradius for each triple
    circumradii = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = pts[i], pts[j], pts[k]
                v1 = a - b
                v2 = b - c
                x = np.cross(v1, v2)
                nx = np.linalg.norm(x)
                if nx < 1e-10:
                    continue
                x = x / nx
                x_neg = -x
                
                for xc in [x, x_neg]:
                    d_i = np.arccos(np.clip(xc @ a, -1, 1))
                    d_j = np.arccos(np.clip(xc @ b, -1, 1))
                    d_k = np.arccos(np.clip(xc @ c, -1, 1))
                    
                    all_dists = np.arccos(np.clip(pts @ xc, -1, 1))
                    min_dist = np.min(all_dists)
                    
                    # Only count if this is a valid Voronoi vertex
                    if (abs(all_dists[i] - min_dist) < 1e-4 and 
                        abs(all_dists[j] - min_dist) < 1e-4 and 
                        abs(all_dists[k] - min_dist) < 1e-4):
                        circumradii.append(min_dist)
    
    if not circumradii:
        return np.pi / 2.0  # fallback
    
    circumradii = np.array(circumradii)
    
    # Soft-max approximation of max
    max_val = np.max(circumradii)
    if max_val < 1e-10:
        return 0.0
    exp_vals = np.exp(-beta * (max_val - circumradii))
    soft_max = max_val + np.log(np.sum(exp_vals)) / beta
    
    return soft_max

def _objective(params):
    """Objective function for optimization: smooth covering radius."""
    pts = _normalize(params.reshape(N, 3))
    # Check validity: no coincident points, not in a hemisphere
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-6):
        return 10.0
    # Check not all in a hemisphere: the origin should be inside the convex hull
    # Simple check: max over directions of min dot product should be < 0
    # Actually, if all points are in a hemisphere, there exists a direction u such
    # that u . p_i > 0 for all i. The origin is in the convex hull iff for all u,
    # min_i u . p_i <= 0.
    # A simpler necessary condition: the sum of points should be near 0 for good coverage.
    center = np.mean(pts, axis=0)
    if np.linalg.norm(center) > 0.5:
        return 10.0 + np.linalg.norm(center)
    
    return _smooth_covering_radius(pts, beta=30.0)

def _objective_grad(params):
    """Gradient of the smooth covering radius objective."""
    pts = _normalize(params.reshape(N, 3))
    n = len(pts)
    beta = 30.0
    
    # Gradient of soft-max over circumradii
    # This is complex; use numerical gradient as fallback
    eps = 1e-5
    grad = np.zeros_like(params)
    f0 = _objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        grad[i] = (f_plus - f0) / eps
    
    return grad

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Evaluate initial covering radius
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Try multiple random restarts and local optimizations
    rng = np.random.RandomState(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if _time_left() < 30.0:
            break
        
        # Create a perturbed starting point
        if restart == 0:
            start_pts = pts.copy()
        else:
            # Random perturbation
            noise = rng.randn(N, 3) * 0.3
            start_pts = pts + noise
            start_pts = _normalize(start_pts)
        
        # Flatten for scipy
        x0 = start_pts.flatten()
        
        # Use L-BFGS-B with the smooth objective
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
        except Exception:
            continue
        
        if _time_left() < 20.0:
            break
        
        opt_pts = _normalize(result.x.reshape(N, 3))
        
        # Evaluate true covering radius
        opt_radius = _covering_radius(opt_pts)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Final polish: high-precision local optimization
    if _time_left() > 15.0:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            opt_pts = _normalize(result.x.reshape(N, 3))
            opt_radius = _covering_radius(opt_pts)
            if opt_radius < best_radius:
                best_radius = opt_radius
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
