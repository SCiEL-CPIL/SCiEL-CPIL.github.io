"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _is_inside(points):
    """Check if all points are inside the equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Point (x,y) is inside if:
    # 1. y >= 0
    # 2. y <= sqrt(3)*x  (below line from (0,0) to (0.5, sqrt(3)/2))
    # 3. y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)  (below line from (1,0) to (0.5, sqrt(3)/2))
    x = points[:, 0]
    y = points[:, 1]
    return np.all(y >= -1e-9) and np.all(y <= np.sqrt(3) * x + 1e-9) and np.all(y <= np.sqrt(3) * (1 - x) + 1e-9)


def _params_to_points(params):
    """Convert 22 parameters to 11 points, ensuring they stay inside the triangle."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        # Map from [0,1]x[0,1] to triangle
        # Using barycentric-like mapping: point = u*A + v*B + (1-u-v)*C
        # But we want to stay inside, so use: if u+v > 1, reflect
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        # Now u,v >= 0 and u+v <= 1
        # Point = u*(0,0) + v*(1,0) + (1-u-v)*(0.5, sqrt(3)/2)
        points[i, 0] = v + 0.5 * (1 - u - v)
        points[i, 1] = H * (1 - u - v)
    return points


def _objective(params):
    """Objective function: negative min area (to minimize)."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_min_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with a symmetric initial guess
    # Place points in a grid-like pattern inside the triangle
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple random restarts with differential evolution
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Random initial population for DE
        bounds = [(0, 1)] * (2 * N)
        
        # Run differential evolution with limited iterations to stay within time
        max_iter = 50 if n_restarts == 1 else 30
        pop_size = 30 if n_restarts == 1 else 20
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-6,
                seed=n_restarts * 100,
                polish=True,
                init='latinhypercube'
            )
        except Exception:
            continue
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x.copy()
        
        # Refine with Nelder-Mead
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    if best_params is None:
        # Fallback: random points
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    points[i, 0] = v + 0.5 * (1 - u - v)
                    points[i, 1] = H * (1 - u - v)
                    break
        return points
    
    points = _params_to_points(best_params)
    
    # Ensure points are inside triangle (clamp if needed)
    for i in range(N):
        x, y = points[i]
        if y < 0:
            y = 0
        if y > np.sqrt(3) * x:
            # Project to line y = sqrt(3)*x
            if x > 0:
                y = np.sqrt(3) * x
            else:
                x, y = 0, 0
        if y > np.sqrt(3) * (1 - x):
            # Project to line y = sqrt(3)*(1-x)
            if x < 1:
                y = np.sqrt(3) * (1 - x)
            else:
                x, y = 1, 0
        points[i] = [x, y]
    
    return points
