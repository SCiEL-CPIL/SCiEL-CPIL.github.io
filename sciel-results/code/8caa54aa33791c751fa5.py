"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Pre-compute triangle indices for speed
def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (n, 2) array
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def smooth_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns -softmin(areas) so that minimizing this maximizes the soft-min.
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp trick for stability
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    softmin = -log_sum / beta + max_a
    return -softmin

def convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    """
    Calculate the score: min_triangle_area / convex_hull_area
    """
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin of 5s before 150s kill
    
    best_score = -1.0
    best_points = None
    
    # Initial guesses
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    rng = np.random.default_rng(42)
    rand_points = rng.random((N, 2)) * 2 - 1
    
    initializations = [reg_points, rand_points]
    for _ in range(5):
        initializations.append(rng.random((N, 2)) * 2 - 1)
    
    def objective_smooth(x, beta=100.0):
        """Smooth objective for gradient-based optimization."""
        pts = x.reshape(-1, 2)
        return smooth_min_triangle_area(pts, beta)
    
    def objective_nonsmooth(x):
        """Nonsmooth objective for local polish."""
        pts = x.reshape(-1, 2)
        return -min_triangle_area(pts)
    
    # Phase 1: Smooth optimization with L-BFGS-B from multiple starts
    for init in initializations:
        if time.time() > deadline - 15.0:
            break
        x0 = init.flatten()
        try:
            res = minimize(objective_smooth, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            pts = res.x.reshape(-1, 2)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish from best result
    if best_points is not None:
        if time.time() < deadline - 10.0:
            x0 = best_points.flatten()
            try:
                res = minimize(objective_nonsmooth, x0, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                pts = res.x.reshape(-1, 2)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass
    
    # Phase 3: High-precision polish with very tight tolerances
    if best_points is not None:
        if time.time() < deadline - 3.0:
            x0 = best_points.flatten()
            try:
                res = minimize(objective_nonsmooth, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
                pts = res.x.reshape(-1, 2)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = initializations[0]
        best_score = score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
