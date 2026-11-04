import numpy as np
import math

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if j + 1 < n:
                pk = points[j+1:]
                v1 = pj - pi
                v2 = pk - pi
                areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
                min_area = min(min_area, np.min(areas))
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull of points using monotone chain."""
    points = np.array(points)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
        
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _normalize(points):
    """Translate to centroid and scale to unit convex hull area."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    area = _convex_hull_area(points)
    if area > 1e-9:
        scale = 1.0 / math.sqrt(area)
        points = points * scale
    return points

def _score_normalized(points):
    """Score is min triangle area, assuming points are normalized to unit hull area."""
    return _min_triangle_area(points)

def _simulated_annealing(points_init, max_iters=30000, T_start=1.0, T_end=1e-6):
    """
    Perform Simulated Annealing to maximize the score on normalized points.
    """
    current_points = _normalize(points_init.copy())
    current_score = _score_normalized(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(123)
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for i in range(max_iters):
        # Generate neighbor
        idx = rng.integers(0, N)
        # Step size proportional to temperature
        step_size = 0.1 * T + 0.01
        delta = rng.normal(0, step_size, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Normalize
        new_points = _normalize(new_points)
        
        new_score = _score_normalized(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            if T > 1e-9:
                delta_score = new_score - current_score
                if delta_score > -10 * T:
                    p = math.exp(delta_score / T)
                    if rng.random() < p:
                        current_points = new_points
                        current_score = new_score
        
        T *= alpha
        
    return best_points, best_score

def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    
    # Initialize with a regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Run Simulated Annealing
    best_points, best_score = _simulated_annealing(points_init, max_iters=30000)
    
    return best_points
