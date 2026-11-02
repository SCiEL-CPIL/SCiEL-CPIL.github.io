"""Heilbronn convex regions n=14: maximize minimum triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # safety margin before 150s kill


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: manual convex hull
        # Find point with lowest y, then lowest x
        start = 0
        for i in range(1, n):
            if points[i][1] < points[start][1] or (points[i][1] == points[start][1] and points[i][0] < points[start][0]):
                start = i
        
        def angle_cmp(p):
            return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
        
        rest = [points[i] for i in range(n) if i != start]
        rest.sort(key=angle_cmp)
        hull_pts = [points[start]] + rest
        
        # Graham scan
        stack = []
        for p in hull_pts:
            while len(stack) >= 2:
                p1 = stack[-2]
                p2 = stack[-1]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross <= 0:
                    stack.pop()
                else:
                    break
            stack.append(p)
        
        hull_pts = stack
        if len(hull_pts) < 3:
            return 0.0
        
        # Shoelace formula
        area = 0.0
        m = len(hull_pts)
        for i in range(m):
            j = (i + 1) % m
            area += hull_pts[i][0] * hull_pts[j][1] - hull_pts[j][0] * hull_pts[i][1]
        return abs(area) / 2.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _objective(points_flat):
    """Objective function for minimization (negative score)."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)


def _initial_grid_config():
    """Generate a reasonable initial configuration using a grid-like pattern."""
    # Use a regular polygon or grid pattern as starting point
    # Try several different initial patterns
    patterns = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    patterns.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two concentric heptagons
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    patterns.append(np.vstack([outer, inner]))
    
    # Grid-like arrangement
    rows, cols = 4, 4  # 16 positions, use 14
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    grid = np.array([[xi, yj] for yj in y for xi in x])
    patterns.append(grid[:N])
    
    # Hexagonal packing approximation
    hex_pts = []
    for i in range(4):
        for j in range(4):
            x = i * 1.0 + (j % 2) * 0.5
            y = j * 0.866
            hex_pts.append([x - 1.5, y - 1.5])
    hex_pts = np.array(hex_pts[:N])
    patterns.append(hex_pts)
    
    return patterns


def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    
    # Get initial configurations
    patterns = _initial_grid_config()
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    
    # Evaluate initial patterns
    for pattern in patterns:
        if time.time() > DEADLINE - 10:
            break
        score = _score(pattern)
        if score > best_score:
            best_score = score
            best_points = pattern.copy()
    
    # Also try random points
    for _ in range(5):
        if time.time() > DEADLINE - 10:
            break
        random_pts = rng.random((N, 2))
        score = _score(random_pts)
        if score > best_score:
            best_score = score
            best_points = random_pts.copy()
    
    if best_points is None:
        # Fallback: return a regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
    
    # Local optimization from the best point
    if time.time() < DEADLINE - 15:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() < DEADLINE - 5:
                candidate = result.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass
    
    # Try more local optimizations from different starting points
    if time.time() < DEADLINE - 20:
        for pattern in patterns[:3]:
            if time.time() > DEADLINE - 10:
                break
            # Add small noise
            noisy = pattern + 0.05 * rng.random((N, 2))
            x0 = noisy.flatten()
            try:
                result = minimize(
                    _objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if time.time() < DEADLINE - 5:
                    candidate = result.x.reshape(-1, 2)
                    cand_score = _score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = candidate
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if time.time() < DEADLINE - 5:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            candidate = result.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
