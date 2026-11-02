"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time
import itertools


def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized computation for speed
    for i in range(n):
        for j in range(i + 1, n):
            # Compute areas for all k > j
            # Triangle area = 0.5 * |cross(pj-pi, pk-pi)|
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]  # shape (n-j-1, 2)
            # Cross product in 2D: v1x*v2y - v1y*v2x
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(-1, 2)
    return -_score_points(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Initial good starting point: regular polygon + center
    # For 13 points, use a 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_init = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts_init = np.vstack([pts_init, center])
    
    # Shuffle to make it less structured
    np.random.seed(42)
    idx = np.random.permutation(N)
    pts_init = pts_init[idx]
    
    best_score = _score_points(pts_init)
    best_pts = pts_init.copy()
    
    # Also try some other structured starts
    # 13 points on a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    score_circle = _score_points(pts_circle)
    if score_circle > best_score:
        best_score = score_circle
        best_pts = pts_circle.copy()
    
    # Grid-like arrangement
    # 4x4 grid minus one corner, then adjust
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_pts.append([i / 3.0, j / 3.0])
    grid_pts = np.array(grid_pts[:N])
    score_grid = _score_points(grid_pts)
    if score_grid > best_score:
        best_score = score_grid
        best_pts = grid_pts.copy()
    
    # Run differential evolution with multiple restarts
    bounds = [(0, 1), (0, 1)] * N
    
    # Use a smaller population and fewer iterations to fit within time
    n_restarts = 0
    max_restarts = 5
    
    while n_restarts < max_restarts:
        if time.time() > deadline:
            break
        
        # For each restart, we run a limited differential evolution
        # We use a subset of the time budget
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Scale maxiter based on remaining time
        maxiter = max(50, int(remaining * 2))
        maxiter = min(maxiter, 300)
        
        # Use a moderate population size
        popsize = max(10, min(20, N // 2 + 5))
        
        try:
            result = differential_evolution(
                _neg_score,
                bounds=bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-4,
                seed=n_restarts * 100 + 42,
                init='sobol',
                polish=False
            )
            
            if result.fun is not None:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = -result.fun
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
            
            # Also check the initial point from this restart
            # DE's internal best might be better captured by checking result.x
        except Exception:
            pass
        
        n_restarts += 1
    
    # Final local refinement using Nelder-Mead on the best point found
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            
            def _neg_score_simple(params):
                pts = params.reshape(-1, 2)
                return -_score_points(pts)
            
            # Try a few local refinements from different starting points
            for i in range(3):
                if time.time() > deadline - 5:
                    break
                
                start = best_pts.copy()
                if i > 0:
                    # Perturb
                    start = start + np.random.normal(0, 0.05, size=start.shape)
                    start = np.clip(start, 0, 1)
                
                result = minimize(
                    _neg_score_simple,
                    start.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                if result.fun is not None:
                    candidate_pts = result.x.reshape(-1, 2)
                    candidate_score = -result.fun
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Return the best points found
    if best_pts is not None and len(best_pts) == N:
        return best_pts
    else:
        # Fallback: return initial points
        return pts_init
# EVOLVE-BLOCK-END
