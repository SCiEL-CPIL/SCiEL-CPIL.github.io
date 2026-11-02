"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 115.0  # safety margin before 120s kill


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective(params):
    """
    Objective for minimization: negative log-barrier penalized min-triangle/hull-area.
    params: 2D array of shape (N, 2) flattened to 1D.
    """
    points = params.reshape(N, 2)
    
    # Check convex hull validity
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e6
    except Exception:
        return 1e6
    
    # Compute min triangle area
    min_tri = _min_triangle_area(points)
    
    # Barrier: penalize points that are too close to hull edges (collinearity)
    # For each point, compute distance to each hull edge; penalize if too small
    hull_vertices = points[hull.vertices]
    n_hull = len(hull_vertices)
    barrier = 0.0
    for i in range(N):
        pi = points[i]
        for j in range(n_hull):
            p1 = hull_vertices[j]
            p2 = hull_vertices[(j + 1) % n_hull]
            # Distance from pi to line through p1, p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx * dx + dy * dy)
            if length < 1e-12:
                continue
            # Signed distance
            dist = abs((pi[0] - p1[0]) * dy - (pi[1] - p1[1]) * dx) / length
            # Only penalize if point is very close to an edge (but not on hull)
            if i not in hull.vertices:
                if dist < 0.01:
                    barrier += (0.01 - dist) ** 2
            else:
                # Hull vertices should be strictly convex
                # Check cross product of adjacent edges
                pass
    
    # Also penalize collinearity of consecutive hull vertices
    for j in range(n_hull):
        p_prev = hull_vertices[(j - 1) % n_hull]
        p_curr = hull_vertices[j]
        p_next = hull_vertices[(j + 1) % n_hull]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 0.005:
            barrier += (0.005 - abs(cross)) ** 2
    
    score = min_tri / hull_area
    return -np.log(score + 1e-10) + 100.0 * barrier


def _make_initial_points(seed):
    """Generate initial points on a circle with some noise."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
    radii = 1.0 + rng.uniform(-0.1, 0.1, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -1.0
    
    # Try multiple starting points
    for seed in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        points = _make_initial_points(seed)
        params = points.flatten()
        
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e5:
                opt_points = result.x.reshape(N, 2)
                s = _score(opt_points)
                if s > best_score:
                    best_score = s
                    best_points = opt_points.copy()
                    
        except Exception:
            continue
    
    # If no good result, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
    
    # Final refinement: local search around best point
    if time.time() - start_time < DEADLINE - 10:
        for trial in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            perturbation = np.random.randn(N, 2) * 0.01
            trial_points = best_points + perturbation
            s = _score(trial_points)
            if s > best_score:
                best_score = s
                best_points = trial_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
