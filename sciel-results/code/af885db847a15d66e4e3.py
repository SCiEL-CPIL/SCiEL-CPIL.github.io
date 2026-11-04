"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _all_triangle_areas_vec(points):
    """Compute areas of all C(n,3) triangles using vectorized operations."""
    n = len(points)
    # Build index arrays for all combinations of 3
    i = np.repeat(np.arange(n), n*(n-1)//2)
    j = np.tile(np.arange(n)[:, None], (1, n-1)).ravel()
    # More careful construction
    ii, jj, kk = np.meshgrid(
        np.arange(n),
        np.arange(n),
        np.arange(n),
        indexing='ij'
    )
    mask = (ii < jj) & (jj < kk)
    ii = ii[mask].astype(int)
    jj = jj[mask].astype(int)
    kk = kk[mask].astype(int)
    
    p1 = points[ii]
    p2 = points[jj]
    p3 = points[kk]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _all_triangle_areas_vec(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return points


def _param_to_points(params):
    """Convert parameter vector to 2D points.
    
    params: array of shape (N, 2) where each row is (log_r, theta)
    """
    n = len(params)
    r = np.exp(params[:, 0])
    theta = params[:, 1]
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return points


def _points_to_param(points):
    """Convert 2D points to parameter vector (log-radii, theta)."""
    n = len(points)
    x, y = points[:, 0], points[:, 1]
    r = np.sqrt(x**2 + y**2)
    r = np.maximum(r, 1e-10)
    theta = np.arctan2(y, x)
    params = np.column_stack([np.log(r), theta])
    return params


def construct():
    start_time = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # Regular 13-gon
    candidates.append(_regular_polygon(N))
    
    # Slightly perturbed regular polygons
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _regular_polygon(N)
        perturbed = base + rng.randn(N, 2) * 0.05
        candidates.append(perturbed)
    
    # Random points in unit disk
    for _ in range(15):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # Points on circle with varied radii (annulus-like)
    for _ in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi/N, 1)
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # Track best
    best_points = None
    best_score = 0.0
    
    for points in candidates:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Local optimization using SLSQP
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        param_init = _points_to_param(best_points)
        
        def objective(params):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                return 0.0
            pts = _param_to_points(params)
            hull_area = _convex_hull_area(pts)
            if hull_area <= 0:
                return 0.0
            areas = _all_triangle_areas_vec(pts)
            if len(areas) == 0:
                return 0.0
            min_area = np.min(areas)
            return -min_area / hull_area
        
        # Multiple restarts with varying perturbation
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            if restart == 0:
                x0 = param_init.copy()
            else:
                # Scale perturbation based on restart count
                scale = 0.05 + 0.02 * restart
                x0 = param_init + rng.randn(N, 2) * scale
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                opt_points = _param_to_points(result.x)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    param_init = result.x.copy()
                    
            except Exception:
                continue
    
    # If we still have time, do a second round of optimization with different method
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        param_init = _points_to_param(best_points)
        
        def objective2(params):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                return 0.0
            pts = _param_to_points(params)
            hull_area = _convex_hull_area(pts)
            if hull_area <= 0:
                return 0.0
            areas = _all_triangle_areas_vec(pts)
            if len(areas) == 0:
                return 0.0
            min_area = np.min(areas)
            return -min_area / hull_area
        
        # Use COBYLA as a derivative-free alternative
        for restart in range(5):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            if restart == 0:
                x0 = param_init.copy()
            else:
                x0 = param_init + rng.randn(N, 2) * 0.03
            
            try:
                result = minimize(
                    objective2,
                    x0,
                    method='COBYLA',
                    options={'maxiter': 300, 'rhobeg': 0.01}
                )
                
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                opt_points = _param_to_points(result.x)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    param_init = result.x.copy()
                    
            except Exception:
                continue
    
    # Final fallback
    if best_points is None:
        best_points = _regular_polygon(N)
    
    return best_points
# EVOLVE-BLOCK-END
