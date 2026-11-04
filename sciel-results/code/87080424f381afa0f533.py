"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11
SQRT3 = np.sqrt(3)
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, SQRT3 / 2.0])
TRIANGLE_AREA = 0.5 * SQRT3 / 2.0  # area of equilateral triangle with side 1

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return 0.5 * abs(cross)

def _min_triangle_area(points):
    """Minimum area of any triangle formed by triples of points."""
    min_a = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area(points, temperature=0.01):
    """Smooth surrogate for minimum triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas.append(a)
    areas = np.array(areas)
    # soft min: -log-sum-exp(-areas/temperature) * temperature
    min_a = np.min(areas)
    if min_a < 1e-15:
        return 0.0
    shifted = -areas / temperature
    max_shifted = np.max(shifted)
    lse = np.log(np.sum(np.exp(shifted - max_shifted))) + max_shifted
    soft_min = -lse * temperature
    return max(0.0, soft_min)

def _is_inside_triangle(pt, tol=1e-9):
    """Check if point is inside or on the boundary of the equilateral triangle."""
    x, y = pt
    if y < -tol:
        return False
    if y > SQRT3 * x + tol:
        return False
    if y > -SQRT3 * (x - 1) + tol:
        return False
    return True

def _project_to_triangle(pt):
    """Project a point to the nearest point on/in the triangle."""
    x, y = pt
    # Check if inside
    if _is_inside_triangle(pt):
        return pt
    # Project to edges
    # Edge AB: y = 0
    # Edge BC: y = -sqrt(3)(x-1) => sqrt(3)x + y - sqrt(3) = 0
    # Edge CA: y = sqrt(3)x => -sqrt(3)x + y = 0
    
    def dist_to_line(p, a, b):
        """Distance from point p to line through a and b."""
        ax, ay = a
        bx, by = b
        px, py = p
        # Line: (by-ay)(x-ax) - (bx-ax)(y-ay) = 0
        A_coef = by - ay
        B_coef = -(bx - ax)
        C_coef = -(by - ay) * ax + (bx - ax) * ay
        dist = abs(A_coef * px + B_coef * py + C_coef) / np.sqrt(A_coef**2 + B_coef**2)
        # Projection
        denom = A_coef**2 + B_coef**2
        t = -(A_coef * px + B_coef * py + C_coef) / denom
        proj_x = px + t * A_coef
        proj_y = py + t * B_coef
        return proj_x, proj_y, dist
    
    candidates = []
    # Project to each edge
    for edge in [(A, B), (B, C), (C, A)]:
        px, py, d = dist_to_line(pt, edge[0], edge[1])
        # Check if projection is within the edge segment
        ax, ay = edge[0]
        bx, by = edge[1]
        t = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
        t = max(0.0, min(1.0, t / ((bx-ax)**2 + (by-ay)**2)))
        proj_x = ax + t * (bx - ax)
        proj_y = ay + t * (by - ay)
        candidates.append((proj_x, proj_y, d))
    
    # Also consider vertices
    for v in [A, B, C]:
        d = np.linalg.norm(np.array(pt) - v)
        candidates.append((v[0], v[1], d))
    
    # Pick closest
    best = min(candidates, key=lambda c: c[2])
    return np.array([best[0], best[1]])

def _flatten(points):
    """Flatten list of 2D points to 1D array."""
    return np.array([coord for pt in points for coord in pt])

def _unflatten(flat):
    """Unflatten 1D array to list of 2D points."""
    return [flat[i:i+2] for i in range(0, len(flat), 2)]

def _objective(flat, temperature=0.01):
    """Objective to minimize: negative soft-min area."""
    points = _unflatten(flat)
    # Penalize points outside triangle
    penalty = 0.0
    for pt in points:
        if not _is_inside_triangle(pt, tol=1e-6):
            # Distance to triangle boundary
            proj = _project_to_triangle(pt)
            penalty += 100.0 * np.linalg.norm(np.array(pt) - proj)
    soft_min = _soft_min_area(points, temperature)
    return -soft_min + penalty

def _gradient(flat, temperature=0.01):
    """Numerical gradient of objective."""
    eps = 1e-7
    grad = np.zeros_like(flat)
    for i in range(len(flat)):
        flat_plus = flat.copy()
        flat_plus[i] += eps
        grad[i] = (_objective(flat_plus, temperature) - _objective(flat, temperature)) / eps
    return grad

def _make_initial_config(seed, strategy='random'):
    """Generate initial configuration of N points."""
    rng = np.random.RandomState(seed)
    points = []
    if strategy == 'random':
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            else:
                # Retry
                u = rng.random()
                v = rng.random()
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            points.append(pt)
    elif strategy == 'grid':
        # Place points in a rough grid pattern
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            else:
                u = rng.random()
                v = rng.random()
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            points.append(pt)
    else:
        # Vertices and midpoints plus random
        points.append(A.copy())
        points.append(B.copy())
        points.append(C.copy())
        points.append((A + B) / 2.0)
        points.append((B + C) / 2.0)
        points.append((C + A) / 2.0)
        for _ in range(N - 6):
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            else:
                u = rng.random()
                v = rng.random()
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            points.append(pt)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Try multiple strategies and seeds
    strategies = ['random', 'grid', 'vertices']
    seeds = list(range(50))
    
    for strategy in strategies:
        for seed in seeds:
            if time.time() > deadline - 10.0:
                break
            
            points = _make_initial_config(seed, strategy)
            flat = _flatten(points)
            
            # Use L-BFGS-B with bounds
            # Bounds: x in [0,1], y in [0, sqrt(3)/2]
            bounds = []
            for _ in range(N):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, SQRT3 / 2.0))
            
            # Try with different temperature values
            for temp in [0.05, 0.02, 0.01]:
                if time.time() > deadline - 5.0:
                    break
                
                try:
                    result = minimize(
                        _objective,
                        flat,
                        args=(temp,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                    )
                    
                    if result.success:
                        opt_points = _unflatten(result.x)
                        # Project to triangle to ensure validity
                        opt_points = [_project_to_triangle(pt) for pt in opt_points]
                        score = _min_triangle_area(opt_points)
                        
                        if score > best_score:
                            best_score = score
                            best_points = opt_points
                except Exception:
                    pass
        
        if time.time() > deadline - 10.0:
            break
    
    # If we have a good result, do a final polish with tighter tolerance
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            flat = _flatten(best_points)
            bounds = []
            for _ in range(N):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, SQRT3 / 2.0))
            
            result = minimize(
                _objective,
                flat,
                args=(0.005,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            
            if result.success:
                opt_points = _unflatten(result.x)
                opt_points = [_project_to_triangle(pt) for pt in opt_points]
                score = _min_triangle_area(opt_points)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_points
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: simple random
        rng = np.random.RandomState(0)
        points = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            else:
                u = rng.random()
                v = rng.random()
                pt = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
            points.append(pt)
        best_points = points
    
    return np.array(best_points)
