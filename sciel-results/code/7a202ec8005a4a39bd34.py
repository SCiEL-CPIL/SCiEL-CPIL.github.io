"""Heilbronn triangles: maximize min triangle area for 11 points in unit triangle."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

N = 11
T_START = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# Pre-compute all triangle index triples
TRI_IDX = np.array(list(combinations(range(N), 3)))

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross).min()

def objective(x):
    """Objective to minimize: negative min area plus boundary penalty."""
    pts = x.reshape(N, 2)
    sqrt3 = np.sqrt(3.0)
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    v1 = np.maximum(0.0, -y_coords)
    v2 = np.maximum(0.0, y_coords - sqrt3 * x_coords)
    v3 = np.maximum(0.0, y_coords - sqrt3 * (1.0 - x_coords))
    v4 = np.maximum(0.0, -x_coords)
    v5 = np.maximum(0.0, x_coords - 1.0)
    penalty = 100.0 * (v1.sum() + v2.sum() + v3.sum() + v4.sum() + v5.sum())
    return -min_triangle_area(pts) + penalty

def clamp_to_triangle(pts):
    """Clamp points to lie inside the equilateral triangle."""
    sqrt3 = np.sqrt(3.0)
    pts = pts.copy()
    for i in range(N):
        x = max(0.0, min(1.0, pts[i, 0]))
        y = max(0.0, pts[i, 1])
        y = min(y, sqrt3 * x)
        y = min(y, sqrt3 * (1.0 - x))
        pts[i] = [x, y]
    return pts

def construct():
    deadline = T_START + TIME_LIMIT
    sqrt3 = np.sqrt(3.0)
    
    # Phase 1: Differential evolution for global search
    bounds = [(0.0, 1.0), (0.0, sqrt3 / 2.0)] * N
    
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=40,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
        )
        best_x = de_result.x.copy()
    except Exception:
        # Fallback: random points
        rng = np.random.RandomState(7)
        best_x = rng.uniform(0.05, 0.95, size=2 * N)
    
    best_val = -objective(best_x)
    
    if time.time() > deadline - 25:
        return clamp_to_triangle(best_x.reshape(N, 2))
    
    # Phase 2: Powell local refinement from multiple starting points
    start_points = [best_x.copy()]
    
    # Perturbed versions of the best
    rng = np.random.RandomState(123)
    for _ in range(8):
        perturbed = best_x + rng.normal(0, 0.03, size=best_x.shape)
        start_points.append(perturbed)
    
    # A few independent random starts
    for seed in range(4):
        rs = np.random.RandomState(seed + 500)
        start_points.append(rs.uniform(0.05, 0.95, size=2 * N))
    
    for sp in start_points:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                objective,
                sp,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-11, 'ftol': 1e-13},
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    pts = clamp_to_triangle(best_x.reshape(N, 2))
    return np.array(pts)
