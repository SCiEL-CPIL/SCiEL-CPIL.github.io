"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
_DEADLINE = 140.0  # Leave 10s margin
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _exact_covering_radius(pts):
    """Compute the exact covering radius in radians.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in pts. This is equivalent to pi - arccos(max_min_dot),
    where max_min_dot is the maximum over all directions y of the minimum dot product
    with any point in pts.
    
    The worst-case direction y lies at a vertex of the Voronoi diagram on the sphere,
    which is the circumcenter of a spherical triangle formed by three points from pts.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Precompute all pairwise dot products? No, we need dot products with circumcenters.
    
    max_dot = -2.0 # Minimum possible dot product is -1, we want to maximize the min dot product
    
    # Iterate over all triples
    # To speed this up, we can skip triples that are likely not Delaunay, but for N=50, full check is fine.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = pts[i], pts[j], pts[k]
                
                # Circumcenter of spherical triangle (a,b,c)
                # The point y on the sphere equidistant to a, b, c is parallel to (a-b) x (b-c)
                v1 = a - b
                v2 = b - c
                y = np.cross(v1, v2)
                ny = np.linalg.norm(y)
                if ny < 1e-10:
                    continue
                y = y / ny
                
                # Check both y and -y
                for yc in [y, -y]:
                    # Compute dot products with all points
                    dots = pts @ yc
                    min_dot = np.min(dots)
                    
                    # For this to be a valid Voronoi vertex determining the covering radius,
                    # the min_dot should be achieved by at least 3 points (or be the global max of mins).
                    # Actually, we just want the global maximum of min_dot over all candidate vertices.
                    # If yc is a Voronoi vertex, min_dot is the value for that vertex.
                    # We take the max over all such vertices.
                    
                    # Optimization: If min_dot is not greater than current max_dot, skip detailed check?
                    # No, we must verify it's a valid Voronoi vertex (empty cap).
                    # A point yc is a Voronoi vertex if the cap centered at yc with radius 
                    # arccos(min_dot) contains no other points. This is equivalent to 
                    # min_dot being the minimum dot product.
                    # So the condition is simply that min_dot is indeed the minimum.
                    # Which is tautologically true by definition of np.min.
                    # However, we must ensure that the three points i, j, k are the ones achieving the min.
                    # If another point m is closer (higher dot product), then yc is not a vertex of the 
                    # Voronoi cell for i,j,k, but it is still a point on the sphere.
                    # The covering radius is the max over ALL y of min_i (y . p_i).
                    # The function f(y) = min_i (y . p_i) is concave? No.
                    # The maximum of f(y) occurs at a vertex of the arrangement of great circles.
                    # These vertices are the circumcenters of triples.
                    # So we just need to evaluate f(y) for all circumcenters of all triples.
                    # f(y) = min_i (y . p_i).
                    # So we just need to compute min_dot for every triple circumcenter and take the max.
                    
                    if min_dot > max_dot:
                        max_dot = min_dot
                        
    # The covering radius is the angle corresponding to this max_dot.
    # max_dot = cos(r) where r is the covering radius.
    # So r = arccos(max_dot).
    return np.arccos(np.clip(max_dot, -1.0, 1.0))

def _fast_objective(params):
    """Fast surrogate for covering radius using a dense grid."""
    pts = _normalize(params.reshape(N, 3))
    
    # Use a pre-computed grid of points on the sphere
    # Global grid to avoid recomputation
    if not hasattr(_fast_objective, 'grid'):
        # Create a Fibonacci spiral of 10,000 points
        M = 10000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid = _normalize(grid)
        _fast_objective.grid = grid
        
    grid = _fast_objective.grid
    
    # Compute dot products: (M, N)
    dots = grid @ pts.T
    # Min over points for each grid point: (M,)
    min_dots = np.min(dots, axis=1)
    # Max over grid points: scalar
    max_min_dot = np.max(min_dots)
    
    # We want to minimize the covering radius, which means maximizing max_min_dot.
    # So objective is -max_min_dot.
    # Add penalty if points are too close (coincident) or in a hemisphere.
    # Check for coincident points
    # This is expensive to do exactly, but we can check min distance between pairs.
    # For N=50, N^2 = 2500.
    # dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    # min_dist = np.min(dists[np.triu_indices(N, k=1)])
    # if min_dist < 1e-3: return 1.0
    
    # Check if in a hemisphere: sum of points should be small.
    center = np.sum(pts, axis=0)
    center_norm = np.linalg.norm(center)
    if center_norm > 0.5:
        return 1.0 + center_norm
        
    return -max_min_dot

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
    
    best_pts = pts.copy()
    # Initial exact radius for tracking
    best_radius = _exact_covering_radius(pts)
    
    rng = np.random.RandomState(42)
    
    # Optimization loop
    # We will try a few restarts if time permits.
    n_restarts = 3
    for restart in range(n_restarts):
        if _time_left() < 30.0:
            break
            
        if restart == 0:
            x0 = pts.flatten()
        else:
            # Perturb the best found so far
            noise = rng.randn(N, 3) * 0.1
            start_pts = best_pts + noise
            start_pts = _normalize(start_pts)
            x0 = start_pts.flatten()
            
        try:
            result = minimize(
                _fast_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
            )
        except Exception:
            continue
            
        if _time_left() < 20.0:
            break
            
        opt_pts = _normalize(result.x.reshape(N, 3))
        
        # Evaluate exact covering radius for the optimized points
        opt_radius = _exact_covering_radius(opt_pts)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
            
    # Final polish with more iterations if time allows
    if _time_left() > 15.0:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _fast_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
            )
            opt_pts = _normalize(result.x.reshape(N, 3))
            opt_radius = _exact_covering_radius(opt_pts)
            if opt_radius < best_radius:
                best_radius = opt_radius
                best_pts = opt_pts.copy()
        except Exception:
            pass
            
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
