import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 14

# Precompute indices for all triangles (i, j, k) with i < j < k
# This avoids generating them in every call.
_indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _indices.append((i, j, k))
_indices = np.array(_indices) # Shape (364, 3)

def _min_triangle_area_over_hull(points):
    points = np.asarray(points, dtype=np.float64).reshape(N, 2)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Vectorized triangle area calculation using precomputed indices
    # points[_indices[:, 0]] gives shape (364, 2)
    p1 = points[_indices[:, 0]]
    p2 = points[_indices[:, 1]]
    p3 = points[_indices[:, 2]]
    
    # Cross product in 2D: (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    # Or determinant:
    # | x1 y1 1 |
    # | x2 y2 1 | = x1(y2-y3) - y1(x2-x3) + (x2*y3 - x3*y2)
    # | x3 y3 1 |
    
    # Using the edge vector method:
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(x):
    """
    Objective function for differential_evolution (minimization).
    """
    score = _min_triangle_area_over_hull(x)
    # We want to maximize score, so minimize -score.
    # If score is 0 (invalid), return a large penalty.
    if score <= 0:
        return 1e6
    return -score

def construct():
    # Bounds for each coordinate.
    # Since the problem is scale invariant, we can restrict the search space.
    # A unit square or circle is sufficient.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Differential Evolution parameters
    # popsize: number of individuals in the population
    # maxiter: maximum number of generations
    # tol: tolerance for termination
    # seed: for reproducibility
    
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=100,
        popsize=15,
        tol=1e-4,
        seed=42,
        polish=False,  # Disable Nelder-Mead polish to save time, DE is usually good enough
        init='latinhypercube' # Better initial spread
    )
    
    best_x = result.x
    return best_x.reshape(N, 2)
