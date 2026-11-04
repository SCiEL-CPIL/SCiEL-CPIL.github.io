"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _all_triangle_areas(points):
    """Compute areas of all C(n,3) triangles formed by points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                areas.append(area)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: use shoelace formula on sorted points if hull fails
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = np.min(_all_triangle_areas(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x, points_2d, temperature=1.0):
    """Soft-min objective for smooth optimization."""
    areas = _all_triangle_areas(points_2d)
    # Use log-sum-exp style soft min
    scaled = areas / temperature
    min_val = np.min(scaled)
    log_sum = min_val + np.log(np.sum(np.exp(scaled - min_val)))
    return -log_sum * temperature  # Negative because we maximize


def _param_to_points(params):
    """Convert parameter vector to 2D points.
    
    params: array of shape (N, 2) where each row is (r, theta)
    We use log-radii for positivity: actual_r = exp(r_param)
    """
    n = len(params)
    points = np.zeros((n, 2))
    for i in range(n):
        r = np.exp(params[i, 0])  # Log-radii parameterization
        theta = params[i, 1]
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
    return points


def _points_to_param(points):
    """Convert 2D points to parameter vector (log-radii, theta)."""
    n = len(points)
    params = np.zeros((n, 2))
    for i in range(n):
        x, y = points[i]
        r = np.sqrt(x**2 + y**2)
        if r < 1e-10:
            r = 1e-10
        theta = np.arctan2(y, x)
        params[i, 0] = np.log(r)
        params[i, 1] = theta
    return params


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return points


def construct():
    start_time = time.time()
    
    # Initial set of candidate configurations
    candidates = []
    
    # Regular 13-gon
    candidates.append(_regular_polygon(N))
    
    # Random initial configurations
    rng = np.random.RandomState(N)
    for _ in range(20):
        points = rng.random((N, 2)) * 2 - 1  # In [-1, 1]^2
        candidates.append(points)
    
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
    
    # Local optimization using SLSQP with log-radii parameterization
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        # Use the best candidate as starting point
        param_init = _points_to_param(best_points)
        
        # Define objective function for optimization
        def objective(params):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                return 0.0
            pts = _param_to_points(params)
            # Normalize by convex hull area
            hull_area = _convex_hull_area(pts)
            if hull_area <= 0:
                return 0.0
            areas = _all_triangle_areas(pts)
            min_area = np.min(areas)
            return -min_area / hull_area  # Negative for minimization
        
        # Try optimization from multiple starting points
        for restart in range(10):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            if restart == 0:
                x0 = param_init.copy()
            else:
                # Perturb the initial parameters
                x0 = param_init + rng.randn(N, 2) * 0.1
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                if result.success:
                    opt_points = _param_to_points(result.x)
                    opt_score = _score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        param_init = result.x.copy()  # Update for next restart
                        
            except Exception:
                continue
    
    # Final check
    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon(N)
    
    return best_points
# EVOLVE-BLOCK-END
