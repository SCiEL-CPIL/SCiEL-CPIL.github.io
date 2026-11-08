"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points from the set.
    Points: (N, 2) array.
    """
    n = len(points)
    # Indices for all triples i < j < k
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return np.min(areas)

def _convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score is min triangle area / convex hull area.
    """
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area == 0:
        return 0.0
    return min_area / hull_area

def construct():
    N = 13
    np.random.seed(42)
    
    # Initialize with a regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.05
    cooling_rate = 0.99
    min_T = 1e-5
    max_iter = 5000
    
    for i in range(max_iter):
        if T < min_T:
            break
            
        # Pick a random point to move
        idx = np.random.randint(N)
        old_point = points[idx].copy()
        
        # Perturb
        step = 0.05 * T + 0.001
        new_point = points[idx] + np.random.normal(0, step, 2)
        
        points[idx] = new_point
        
        new_score = _score(points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_score = new_score
                else:
                    points[idx] = old_point
            else:
                current_score = new_score
        
        T *= cooling_rate
        
    return best_points
# EVOLVE-BLOCK-END
