# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # Safety margin before 630s limit

# Precompute triangle indices for fast scoring
_N = 13
_TRI_INDICES = np.array(list(itertools.combinations(range(_N), 3)))

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed using precomputed indices.
    """
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _objective(x):
    """
    Objective function to minimize (negative of the score).
    """
    points = x.reshape(_N, 2)
    return -_score(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Define bounds for Differential Evolution
    # Normalizing to [0, 1] helps DE explore the space effectively
    # The ratio is scale-invariant, so the absolute scale doesn't matter, 
    # but bounding prevents numerical issues and keeps points in a manageable region.
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Stage 1: Global Search with Differential Evolution
    # DE is robust for non-smooth, multimodal functions.
    # We limit the population size and max iterations to fit within time budget.
    # 'maxiter' and 'popsize' control the computational cost.
    # 100 iterations with popsize 20 (approx 2000 evaluations) is a reasonable start.
    # We can increase if time allows, but 2000 evals of 286 triangles is fast enough.
    
    try:
        de_res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            polish=False  # We will do our own polish
        )
        
        best_points = de_res.x.reshape(n, 2)
        best_score = -de_res.fun
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random if DE fails
        rng = np.random.default_rng(seed=42)
        best_points = rng.random((n, 2))
        best_score = _score(best_points)

    # Stage 2: Local Refinement with Powell
    # Powell is good for non-smooth functions and doesn't require gradients.
    if time.time() < _DEADLINE - 60:
        try:
            res = minimize(
                _objective, 
                best_points.flatten(), 
                method='Powell', 
                options={
                    'maxiter': 5000, 
                    'xtol': 1e-10, 
                    'ftol': 1e-12,
                    'niter': 100
                }
            )
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(n, 2)
                
        except Exception as e:
            print(f"Powell failed: {e}")

    # Stage 3: Additional Local Refinement if time permits
    # Try Nelder-Mead or another Powell run from the best point
    if best_points is not None and time.time() < _DEADLINE - 30:
        try:
            res2 = minimize(
                _objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            refined_score = -res2.fun
            if refined_score > best_score:
                best_score = refined_score
                best_points = res2.x.reshape(n, 2)
        except Exception as e:
            print(f"NM failed: {e}")

    if best_points is None:
        # Final fallback
        rng = np.random.default_rng(seed=42)
        best_points = rng.random((n, 2))
        
    return best_points

# EVOLVE-BLOCK-END
