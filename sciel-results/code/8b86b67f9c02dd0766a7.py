"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    # Shoelace formula
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) / 2.0

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) 
                             - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                areas[i, j, k] = a
    return areas

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) 
                             - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if a < min_area:
                    min_area = a
    return min_area

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_ta = _min_triangle_area(points)
    return min_ta / hull_area

def _soft_min_triangle_area(points, temperature=0.01):
    """Soft-min of all triangle areas using log-sum-exp."""
    n = len(points)
    # Vectorized computation of all triangle areas
    # For efficiency, compute all cross products
    min_val = float('inf')
    sum_exp = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) 
                             - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if a < min_val:
                    min_val = a
                sum_exp += np.exp(-a / temperature)
                count += 1
    # soft-min = -temperature * ln(sum(exp(-a/temperature)))
    soft_min = -temperature * np.log(sum_exp / count)
    return soft_min

def _soft_min_grad(points, temperature=0.01):
    """Gradient of soft-min triangle area w.r.t. points."""
    n = len(points)
    grad = np.zeros_like(points)
    # For each triangle, compute gradient of -a w.r.t. points
    # a = 0.5 * |cross(pj-pi, pk-pi)|
    # We need d(a)/d(pi), d(a)/d(pj), d(a)/d(pk)
    # Let's compute for each triangle
    weights = np.zeros((n, n, n))
    sum_exp = 0.0
    count = 0
    tri_areas = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) 
                             - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                w = np.exp(-a / temperature)
                weights[i, j, k] = w
                sum_exp += w
                count += 1
                tri_areas[(i, j, k)] = (a, w)
    
    norm = sum_exp / count
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, w = tri_areas[(i, j, k)]
                # a = 0.5 * |cross(v1, v2)| where v1 = pj - pi, v2 = pk - pi
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                sign = 1.0 if cross >= 0 else -1.0
                # da/d(pi) = -0.5 * sign * (v1[1], -v1[0]) ... wait
                # a = 0.5 * sign * (v1[0]*v2[1] - v1[1]*v2[0])
                # da/dpi_x = 0.5*sign*(v2[1] - v1[1]) ... let me be careful
                # v1 = (pj_x - pi_x, pj_y - pi_y), v2 = (pk_x - pi_x, pk_y - pi_y)
                # cross = (pj_x-pi_x)(pk_y-pi_y) - (pj_y-pi_y)(pk_x-pi_x)
                # d(cross)/d(pi_x) = -(pk_y-pi_y) + (pj_y-pi_y) = v1[1] - v2[1]
                # d(cross)/d(pi_y) = (pk_x-pi_x) - (pj_x-pi_x) = v2[0] - v1[0]
                # d(cross)/d(pj_x) = (pk_y-pi_y) = v2[1]
                # d(cross)/d(pj_y) = -(pk_x-pi_x) = -v2[0]
                # d(cross)/d(pk_x) = -(pj_y-pi_y) = -v1[1]
                # d(cross)/d(pk_y) = (pj_x-pi_x) = v1[0]
                coeff = 0.5 * sign * w / norm
                grad[i] += -coeff * np.array([v1[1] - v2[1], v2[0] - v1[0]])
                grad[j] += -coeff * np.array([v2[1], -v2[0]])
                grad[k] += -coeff * np.array([-v1[1], v1[0]])
    return grad

def _optimize_from_start(start_points, max_iter=200, timeout=10.0):
    """Optimize from a starting configuration."""
    start_time = time.time()
    points = np.array(start_points, dtype=float)
    best_points = points.copy()
    best_score = _score(points)
    
    temperature = 0.02
    
    def objective(x):
        pts = x.reshape(N, 2)
        return _soft_min_triangle_area(pts, temperature)
    
    def grad(x):
        pts = x.reshape(N, 2)
        g = _soft_min_grad(pts, temperature)
        return g.flatten()
    
    x0 = points.flatten()
    
    for temp_iter in range(5):
        if time.time() - start_time > timeout:
            break
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B', 
                         options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() - start_time > timeout:
            break
        x_opt = result.x.reshape(N, 2)
        score = _score(x_opt)
        if score > best_score:
            best_score = score
            best_points = x_opt.copy()
        # Anneal temperature
        temperature *= 0.5
        x0 = x_opt
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial guess: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Evaluate regular polygon
    score = _score(regular)
    if score > best_score:
        best_score = score
        best_points = regular.copy()
    
    # Try optimization from regular polygon
    if time.time() < deadline - 15:
        pts, sc = _optimize_from_start(regular, max_iter=200, timeout=12.0)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Try several random restarts
    rng = np.random.RandomState(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        # Random points in a disk
        theta = rng.uniform(0, 2 * np.pi, N)
        rho = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        
        # Optimize from this start
        if time.time() < deadline - 12:
            opt_pts, opt_sc = _optimize_from_start(pts, max_iter=100, timeout=5.0)
            if opt_sc > best_score:
                best_score = opt_sc
                best_points = opt_pts.copy()
    
    # Final polish: try to improve with more iterations if time allows
    if best_points is not None and time.time() < deadline - 5:
        final_pts, final_sc = _optimize_from_start(best_points, max_iter=300, timeout=8.0)
        if final_sc > best_score:
            best_score = final_sc
            best_points = final_pts.copy()
    
    # Normalize: scale so convex hull area = 1
    if best_points is not None:
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-15:
            scale = 1.0 / np.sqrt(hull_area)
            best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
