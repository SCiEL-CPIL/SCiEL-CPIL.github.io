"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Pre-compute all triangle index combinations for vectorization
_TRI_INDICES = np.array(list(combinations(range(N), 3)))

def _min_triangle_area_vec(points):
    """
    Compute the minimum area over all triples of points.
    Vectorized for speed.
    """
    if len(points) < 3:
        return 0.0
    # Extract coordinates for each triangle vertex
    pi = points[_TRI_INDICES[:, 0]]
    pj = points[_TRI_INDICES[:, 1]]
    pk = points[_TRI_INDICES[:, 2]]
    
    # Compute cross product for area
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area_fast(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """
    Score = min triangle area / convex hull area.
    Higher is better.
    """
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vec(points)
    return min_tri / hull_area

def _objective_free(flat_points):
    """
    Objective function for free 2D points.
    Minimizes negative score.
    """
    points = flat_points.reshape(-1, 2)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial guess: Regular 14-gon on unit circle
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    initial_score = _score(initial_points)
    
    best_score_val = initial_score
    best_points = initial_points.copy()
    
    # Use Differential Evolution for global optimization
    # Bounds: points can move within a reasonable range around the origin
    # The hull area normalization makes the problem scale-invariant,
    # so we can constrain the search space without loss of generality.
    
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    try:
        res = differential_evolution(
            _objective_free,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-10,
            popsize=15,
            polish=True,
            init='sobol',
            workers=1
        )
        
        if res.fun < 0:
            current_score = -res.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_points = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Final sanity check: ensure points are valid
    if best_points is None or len(best_points) != N:
        best_points = initial_points

    return best_points

# EVOLVE-BLOCK-END
