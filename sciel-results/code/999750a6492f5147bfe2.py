"""Heilbronn triangles: maximize the smallest triangle area among 11 points in a unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
SQRT3 = 3 ** 0.5
H = SQRT3 / 2.0

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _in_triangle(pt):
    """Check if point is inside the equilateral triangle (with small margin)."""
    x, y = pt
    if y < -1e-9 or y > H + 1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    # Check against the three edges
    # Edge V0-V1: y >= 0 (already checked)
    # Edge V1-V2: line from (1,0) to (0.5, H)
    # The line equation: y = -sqrt(3)(x - 1) => sqrt(3)*x + y - sqrt(3) <= 0
    if SQRT3 * x + y - SQRT3 > 1e-9:
        return False
    # Edge V0-V2: line from (0,0) to (0.5, H)
    # The line equation: y = sqrt(3)*x => y - sqrt(3)*x >= 0
    if y - SQRT3 * x < -1e-9:
        return False
    return True


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    return 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


def _min_triangle_area(points):
    """Minimum area of any triangle formed by three of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _objective_flat(flat_params):
    """Objective for differential_evolution: minimize negative min area.
    flat_params is a 22-element array (11 points x 2 coords)."""
    pts = flat_params.reshape(N, 2)
    # Check all points are inside the triangle
    for pt in pts:
        if not _in_triangle(pt):
            return 1.0  # Large penalty
    min_a = _min_triangle_area(pts)
    return -min_a


def _refine_local(pts, time_limit):
    """Local refinement using Nelder-Mead on the 22-d parameter vector."""
    from scipy.optimize import minimize
    start_time = time.time()
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    
    def obj(flat):
        p = flat.reshape(N, 2)
        for pt in p:
            if not _in_triangle(pt):
                return 1.0
        return -_min_triangle_area(p)
    
    x0 = best_pts.flatten()
    res = minimize(obj, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    if res.x is not None and res.fun < 0:
        new_pts = res.x.reshape(N, 2)
        if all(_in_triangle(pt) for pt in new_pts):
            new_val = _min_triangle_area(new_pts)
            if new_val > best_val:
                best_pts = new_pts
                best_val = new_val
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Initial diverse starting points
    rng = np.random.default_rng(42)
    
    # Start with a reasonable initial configuration
    # Place points roughly in a grid-like pattern inside the triangle
    def _initial_config(seed):
        r = np.random.default_rng(seed)
        pts = []
        # Try to spread points
        for _ in range(N):
            for _attempt in range(1000):
                u, v = r.random(), r.random()
                if u + v < 1.0:
                    p = [u + 0.5 * v, H * v]
                    # Check distance from existing points
                    if all(np.linalg.norm(np.array(p) - np.array(q)) > 0.05 for q in pts):
                        pts.append(p)
                        break
            if len(pts) < N:
                pts.append(p if 'p' in dir() else [0.5, H/3])
        return np.array(pts[:N])
    
    best_pts = _initial_config(42)
    best_val = _min_triangle_area(best_pts)
    
    # Differential evolution with multiple restarts
    bounds = [(-0.1, 1.1), (-0.1, H + 0.1)] * N
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        try:
            result = differential_evolution(
                _objective_flat, bounds,
                seed=restart * 100 + 7,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                init='sobol',
                polish=False,
                update_interval=10
            )
            
            if result.x is not None and result.fun < 0:
                candidate = result.x.reshape(N, 2)
                if all(_in_triangle(pt) for pt in candidate):
                    val = _min_triangle_area(candidate)
                    if val > best_val:
                        best_val = val
                        best_pts = candidate
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # Local refinement
    if time.time() < deadline - 5:
        best_pts = _refine_local(best_pts, deadline - time.time())
        best_val = _min_triangle_area(best_pts)
    
    return best_pts
