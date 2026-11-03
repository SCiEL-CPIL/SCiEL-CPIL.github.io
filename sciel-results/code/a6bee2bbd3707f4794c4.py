"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3 = 3 ** 0.5

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _all_triangle_areas(points):
    """Compute all triangle areas for all triples of points."""
    n = len(points)
    areas = []
    for i, j, k in combinations(range(n), 3):
        areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    return np.min(_all_triangle_areas(points))

def _smooth_min_area(points, alpha=50.0):
    """Smooth approximation of the minimum triangle area using log-sum-exp."""
    areas = _all_triangle_areas(points)
    # Use -logsumexp(-areas/alpha) as smooth min approximation
    # This is differentiable and approximates min when alpha is small
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-shifted / alpha)))
    return max_a - alpha * log_sum

def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    max_y = S3 / 2 * min(x, 1 - x)
    if y > max_y + eps:
        return False
    return True

def _project_to_triangle(p):
    """Project a point onto the equilateral triangle."""
    x, y = p
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    max_y = S3 / 2 * min(x, 1 - x)
    if y > max_y:
        y = max_y
    return np.array([x, y])

def _penalty(params):
    """Penalty for points outside the triangle."""
    points = params.reshape(N, 2)
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        if y < 0:
            penalty += abs(y)
        if x < 0:
            penalty += abs(x)
        if x > 1:
            penalty += abs(x - 1)
        max_y = S3 / 2 * min(max(x, 0), 1 - max(x, 0))
        if y > max_y:
            penalty += abs(y - max_y)
    return penalty

def _objective(params, alpha=50.0):
    """Objective function: negative smooth min area + penalty for out-of-triangle."""
    points = params.reshape(N, 2)
    pen = _penalty(params)
    if pen > 1e-6:
        return 100.0 + pen * 100
    return -_smooth_min_area(points, alpha)

def _local_optimize(params, alpha_start=100.0, alpha_end=10.0, n_anneal_steps=5):
    """Anneal alpha from large to small, running SLSQP at each step."""
    best_params = params.copy()
    best_val = _min_triangle_area(params.reshape(N, 2))
    
    alphas = np.linspace(alpha_start, alpha_end, n_anneal_steps)
    
    for alpha in alphas:
        def obj(p):
            return _objective(p, alpha)
        
        constraints = []
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: S3/2 * p[2*i] - p[2*i+1]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: S3/2 * (1 - p[2*i]) - p[2*i+1]})
        
        bounds = [(0, 1)] * (2 * N)
        
        try:
            result = minimize(
                obj, best_params, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            points_res = result.x.reshape(N, 2)
            valid = all(_in_triangle(points_res[i], eps=1e-5) for i in range(N))
            if valid:
                val = _min_triangle_area(points_res)
                if val > best_val:
                    best_val = val
                    best_params = result.x.copy()
        except Exception:
            pass
    
    return best_params, best_val

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Config 1: Grid-like with vertices and midpoints
    pts = []
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, S3/2])
    pts.append([0.5, 0.0])
    pts.append([0.25, S3/4])
    pts.append([0.75, S3/4])
    for _ in range(5):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        b1, b2, b3 = 1 - u - v, u, v
        x = b2 * 1 + b3 * 0.5
        y = b3 * S3 / 2
        pts.append([x, y])
    configs.append(np.array(pts))
    
    # Config 2: Random uniform in triangle
    for _ in range(3):
        pts = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            b1, b2, b3 = 1 - u - v, u, v
            x = b2 * 1 + b3 * 0.5
            y = b3 * S3 / 2
            pts.append([x, y])
        configs.append(np.array(pts))
    
    # Config 3: Hexagonal/symmetric pattern
    pts = []
    cx, cy = 0.5, S3/6
    pts.append([cx, cy])
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        r = 0.18
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        if _in_triangle([x, y]):
            pts.append([x, y])
    while len(pts) < N:
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        b1, b2, b3 = 1 - u - v, u, v
        x = b2 * 1 + b3 * 0.5
        y = b3 * S3 / 2
        pts.append([x, y])
    configs.append(np.array(pts[:N]))
    
    # Config 4: Known good pattern (approximate)
    # Based on literature values for n=11
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3/2],
        [0.5, 0.0],
        [0.25, S3/4],
        [0.75, S3/4],
        [0.5, S3/6],
        [0.125, S3/8],
        [0.875, S3/8],
        [0.375, 3*S3/8],
        [0.625, 3*S3/8],
    ]
    configs.append(np.array(pts))
    
    return configs

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    np.random.seed(42)
    
    best_points = None
    best_val = -float('inf')
    
    # Phase 1: Global search with differential evolution
    bounds = [(0, 1)] * (2 * N)
    
    def de_obj(params):
        points = params.reshape(N, 2)
        pen = _penalty(params)
        if pen > 1e-4:
            return 100.0 + pen
        return -_min_triangle_area(points)
    
    try:
        de_result = differential_evolution(
            de_obj, bounds, seed=42, maxiter=100, popsize=15,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
            polish=False, init='sobol'
        )
        de_points = de_result.x.reshape(N, 2)
        de_val = _min_triangle_area(de_points)
        if all(_in_triangle(de_points[i], eps=1e-5) for i in range(N)):
            best_points = de_points
            best_val = de_val
    except Exception:
        pass
    
    # Phase 2: Local optimization from multiple starts
    configs = _generate_initial_configs()
    
    # Add DE result if valid
    if best_points is not None:
        configs.append(best_points.copy())
    
    for config in configs:
        try:
            params, val = _local_optimize(config.flatten())
            if val > best_val:
                best_val = val
                best_points = params.reshape(N, 2)
        except Exception:
            pass
    
    # Phase 3: Final polish with very small alpha
    if best_points is not None:
        try:
            params, val = _local_optimize(best_points.flatten(), alpha_start=20.0, alpha_end=5.0, n_anneal_steps=3)
            if val > best_val:
                best_val = val
                best_points = params.reshape(N, 2)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = configs[0]
    
    # Ensure all points are in triangle
    for i in range(N):
        best_points[i] = _project_to_triangle(best_points[i])
    
    return best_points
