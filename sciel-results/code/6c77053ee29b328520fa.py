"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 14

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import minimize


def _triangle_areas(points):
    """Compute areas of all C(n,3) triangles."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                areas.append(area)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute convex hull area using scipy if available, else manual."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is area
    except Exception:
        # Fallback: compute hull manually
        n = len(points)
        # Find leftmost point
        idx = np.argmin(points[:, 0])
        hull_points = [points[idx]]
        current = idx
        for _ in range(n):
            # Find next point in hull
            best_angle = -np.inf
            best_idx = -1
            for j in range(n):
                if j == current:
                    continue
                v1 = points[current] - points[hull_points[-1] if len(hull_points) > 1 else points[idx]]
                v2 = points[j] - points[current]
                # Cross product to determine turn
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross > 0:
                    angle = np.arctan2(cross, np.dot(v1, v2))
                    if angle > best_angle:
                        best_angle = angle
                        best_idx = j
            if best_idx == -1:
                break
            current = best_idx
            if current == idx:
                break
            hull_points.append(points[current])
        hull_points = np.array(hull_points)
        if len(hull_points) < 3:
            return 0.0
        # Shoelace formula
        x = hull_points[:, 0]
        y = hull_points[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area


def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return np.min(areas) / hull_area


def _soft_min_objective(params, n, temperature):
    """Soft-min of triangle areas (negative for minimization)."""
    points = params.reshape(n, 2)
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    normalized = areas / hull_area
    # Soft min via log-sum-exp
    min_val = np.min(normalized)
    # Use log-sum-exp for stability
    shifted = normalized - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature))) + min_val / temperature
    soft_min = temperature * (-np.log(np.sum(np.exp(-normalized / temperature))))
    return -soft_min


def _hard_objective(params, n):
    """Hard min triangle area / hull area (negative for minimization)."""
    points = params.reshape(n, 2)
    score = _score(points)
    return -score


def _normalize_points(points):
    """Normalize points to unit area convex hull."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = np.sqrt(1.0 / hull_area)
    return points * scale


def _initial_points(rng):
    """Generate initial points in a unit square."""
    points = np.array([[rng.random(), rng.random()] for _ in range(N)])
    return points


def _refine(points, max_iter=1000, tol=1e-12):
    """Refine points using L-BFGS-B on the hard objective."""
    x0 = points.flatten()
    
    def obj(x):
        return _hard_objective(x, N)
    
    result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    return result.x.reshape(N, 2)


def _refine_soft(points, max_iter=2000, temperature=0.01):
    """Refine points using L-BFGS-B on the soft-min objective."""
    x0 = points.flatten()
    
    def obj(x):
        return _soft_min_objective(x, N, temperature)
    
    result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    return result.x.reshape(N, 2)


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_points = None
    best_score = 0.0
    
    rng = random.Random(42)
    
    # Multiple restarts with different initializations
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial points
        if restart == 0:
            # Start with a regular grid-like pattern
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.sqrt(np.linspace(0.1, 1.0, N))
            points = np.column_stack([0.5 + 0.4 * radii * np.cos(angles), 
                                       0.5 + 0.4 * radii * np.sin(angles)])
        elif restart == 1:
            # Start with points on a circle
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            points = np.column_stack([0.5 + 0.45 * np.cos(angles), 
                                       0.5 + 0.45 * np.sin(angles)])
        else:
            # Random initialization
            points = np.array([[0.1 + 0.8 * rng.random(), 0.1 + 0.8 * rng.random()] for _ in range(N)])
        
        # Phase 1: Soft-min refinement with annealing
        for temp in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 15.0:
                break
            points = _refine_soft(points, max_iter=500, temperature=temp)
        
        # Phase 2: Hard objective polish
        if time.time() > deadline - 5.0:
            break
        points = _refine(points, max_iter=2000, tol=1e-14)
        
        # Evaluate
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final high-precision polish if we have time
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _refine(best_points, max_iter=5000, tol=1e-15)
        best_score = _score(best_points)
    
    if best_points is None:
        # Fallback: random points
        best_points = np.array([[rng.random(), rng.random()] for _ in range(N)])
    
    return best_points
# EVOLVE-BLOCK-END
