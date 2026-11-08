import numpy as np

N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by 3 points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _objective(x):
    """Objective function to minimize (negative score)."""
    points = x.reshape(N, 2)
    score = _score(points)
    return -score

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to hull area.
    Uses local optimization starting from regular polygon.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline for safety
    
    # Start with regular 13-gon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Try local optimization with Nelder-Mead
    x0 = best_points.flatten()
    
    # Run optimization
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if time.time() < deadline:
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # If we have time, try a few more restarts with perturbed initial conditions
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() >= deadline - 20:
                break
            
            # Perturb the best solution
            perturbation = rng.normal(0, 0.1, (N, 2))
            x0 = (best_points + perturbation).flatten()
            
            try:
                result = minimize(_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
