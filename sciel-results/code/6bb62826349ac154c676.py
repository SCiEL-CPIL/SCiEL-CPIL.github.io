"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools


def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                # Area = 0.5 * |cross product|
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                areas.append(area)
    return np.array(areas)


def _score(pts):
    """Compute the minimum triangle area divided by the convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0


def _negative_score(pts_flat):
    """Objective function for minimization (negative score)."""
    n = len(pts_flat) // 2
    pts = pts_flat.reshape(n, 2)
    return -_score(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with regular polygon as initial guess
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_score = _score(initial_pts)
    best_pts = initial_pts.copy()
    best_score = initial_score
    
    # Use differential evolution for global optimization
    # We optimize all 2N coordinates
    bounds = [(-1, 1)] * (2 * N)
    
    # Run differential evolution with limited iterations
    def de_objective(x):
        if time.time() > deadline:
            return 0.0
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=10,
            polish=False,
            callback=lambda x, convergence: None
        )
        if result.success and -result.fun > best_score:
            best_pts = result.x.reshape(N, 2)
            best_score = -result.fun
    except Exception:
        pass
    
    # Refine with local optimization if time permits
    if time.time() < deadline - 30:
        try:
            def local_obj(x):
                if time.time() > deadline:
                    return 0.0
                pts = x.reshape(N, 2)
                return -_score(pts)
            
            result2 = minimize(
                local_obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if -result2.fun > best_score:
                best_pts = result2.x.reshape(N, 2)
                best_score = -result2.fun
        except Exception:
            pass
    
    # Further refinement
    if time.time() < deadline - 10:
        try:
            def local_obj2(x):
                if time.time() > deadline:
                    return 0.0
                pts = x.reshape(N, 2)
                return -_score(pts)
            
            result3 = minimize(
                local_obj2,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if -result3.fun > best_score:
                best_pts = result3.x.reshape(N, 2)
                best_score = -result3.fun
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
