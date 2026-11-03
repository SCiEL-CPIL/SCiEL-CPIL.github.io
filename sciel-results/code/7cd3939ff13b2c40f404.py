"""
Rewritten program for Heilbronn convex regions n=14.
Strategy:
1. Assume optimal points lie on the boundary of the convex region.
2. Parameterize the boundary as a unit circle.
3. Optimize the 14 angular positions to maximize the minimum triangle area 
   divided by the convex hull area (which is the area of the inscribed polygon).
4. Use Differential Evolution for global search and Nelder-Mead for local polish.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _points_from_angles(angles):
    """Convert angles to points on the unit circle."""
    angles = np.asarray(angles)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        # ConvexHull requires at least 3 non-collinear points
        if len(points) < 3:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    min_area = float('inf')
    # Vectorized calculation of all triangle areas
    # We can use broadcasting to compute areas for all triples
    # i, j, k with i < j < k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # To be efficient, we iterate over the first point i
    for i in range(n):
        pi = points[i]
        others = np.delete(points, i, axis=0)
        # Shift others so pi is at origin
        v1 = others[:, 0] - pi[0]
        v2 = others[:, 1] - pi[1]
        # Cross product for all pairs in 'others'
        # cross[i, j] = v1[i]*v2[j] - v2[i]*v1[j]
        cross = np.abs(np.outer(v1, v2) - np.outer(v2, v1))
        # We only need unique pairs from 'others', so upper triangle
        idx = np.triu_indices(len(others), k=1)
        areas = 0.5 * cross[idx]
        if len(areas) > 0:
            min_in_i = np.min(areas)
            if min_in_i < min_area:
                min_area = min_in_i
    return min_area

def _score_from_points(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _objective(angles):
    """
    Objective function for optimization (minimization).
    We minimize the negative of the score.
    """
    points = _points_from_angles(angles)
    score = _score_from_points(points)
    return -score

def construct():
    """
    Construct 14 points to maximize min triangle area / hull area.
    Returns array of shape (14, 2).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial guess: Regular 14-gon
    init_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Bounds for angles: [0, 2*pi]
    bounds = [(0, 2 * np.pi)] * N

    best_score = _score_from_points(_points_from_angles(init_angles))
    best_angles = init_angles.copy()

    # Strategy 1: Differential Evolution
    # DE is robust for global optimization in moderate dimensions.
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            polish=False, # We will polish separately
            init='sobol',
            updating='deferred',
            workers=1
        )
        
        if time.time() < deadline - 20:
            candidate_angles = result.x
            candidate_score = -_objective(candidate_angles)
            if candidate_score > best_score:
                best_score = candidate_score
                best_angles = candidate_angles
    except Exception:
        pass

    # Strategy 2: Local Polish with Nelder-Mead
    # NM is good for refining local optima.
    if time.time() < deadline - 10:
        try:
            result2 = minimize(
                _objective,
                best_angles,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            candidate_angles = result2.x
            candidate_score = -_objective(candidate_angles)
            if candidate_score > best_score:
                best_score = candidate_score
                best_angles = candidate_angles
        except Exception:
            pass

    # Strategy 3: Multi-start Local Search
    # If time remains, try a few random restarts with local polish.
    if time.time() < deadline - 15:
        rng = np.random.default_rng(123)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            rand_angles = rng.uniform(0, 2 * np.pi, N)
            
            # Quick polish
            try:
                res = minimize(
                    _objective,
                    rand_angles,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                candidate_angles = res.x
                candidate_score = -_objective(candidate_angles)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_angles = candidate_angles
            except Exception:
                pass

    # Final Polish
    if time.time() < deadline - 5:
        try:
            result3 = minimize(
                _objective,
                best_angles,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            candidate_angles = result3.x
            candidate_score = -_objective(candidate_angles)
            if candidate_score > best_score:
                best_score = candidate_score
                best_angles = candidate_angles
        except Exception:
            pass

    # Convert best angles to Cartesian points
    best_points = _points_from_angles(best_angles)
    
    return best_points
