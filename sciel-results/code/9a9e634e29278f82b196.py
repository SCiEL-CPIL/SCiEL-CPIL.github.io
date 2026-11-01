"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    # Iterate over all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 1e-10 # Avoid division by zero if points are collinear/degenerate

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    return min_area / hull_area

def construct():
    # Initialize with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.99
    max_iters = 50000 # Adjust based on time limit
    start_time = time.time()
    time_limit = 100 # seconds
    
    current_points = best_points.copy()
    current_score = best_score
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random point to perturb
        idx = np.random.randint(N)
        # Random perturbation
        delta = np.random.normal(0, T, 2)
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp((new - old) / T)
            if np.random.random() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_points
# EVOLVE-BLOCK-END
