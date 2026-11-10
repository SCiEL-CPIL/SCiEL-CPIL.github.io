import numpy as np
from scipy.spatial import ConvexHull
import itertools

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 0.0
        if a_hull < 1e-10:
            return 0.0
        
        # Vectorized calculation of all triangle areas
        # Generate indices for all pairs (i, j) with i < j
        # Then for each k > j, compute area
        # To keep it simple and fast enough for n=13, we can use broadcasting or a small loop
        # Let's try a semi-vectorized approach:
        # For each i, j, k, area is 0.5 * |det(p2-p1, p3-p1)|
        # This might be overkill. 286 triangles is small.
        # Let's stick to a fast Python loop but optimized, or use itertools
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
        if min_area < 1e-10:
            return 0.0
        return min_area / a_hull

    rng = np.random.default_rng(seed=42)
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 0.001
    alpha = 0.999 # Cooling rate
    max_iter = 5000
    best_points = None
    best_score = -np.inf
    
    # Multiple restarts
    for restart in range(10):
        if restart == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 1:
            # Jittered Grid
            # 4x4 grid minus one corner, scaled
            grid = np.array([[i, j] for i in range(4) for j in range(4)])
            # Take first 13
            points = grid[:13].astype(float)
            points += rng.normal(0, 0.1, points.shape)
            # Center and scale
            points -= points.mean(axis=0)
            points /= np.max(np.abs(points))
        else:
            # Random points in unit disk
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = np.sqrt(rng.uniform(0, 1, n))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        current_score = get_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        T = T_start
        for i in range(max_iter):
            # Select a random point to move
            idx = rng.integers(n)
            orig = points[idx].copy()
            
            # Step size decreases with temperature
            step_size = 0.01 * (1 + T / T_start)
            points[idx] += rng.normal(0, step_size, 2)
            
            new_score = get_score(points)
            
            # Acceptance criterion
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Accept worse solution with probability exp(-delta / T)
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_score = new_score
                    else:
                        points[idx] = orig
                else:
                    # This case (new_score > current_score) is handled above, 
                    # but if delta < 0 it means new is better, which is also handled.
                    # If delta == 0, accept?
                    pass
            
            T *= alpha
            if T < T_end:
                break
                
    return best_points
