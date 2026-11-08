"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Optimization: We can precompute pairwise distances or use vectorized ops, 
    # but for N=13, 286 triples is small.
    # However, we can speed this up by checking only pairs that are close.
    # For now, keep it simple but efficient in Python loops is fine for 286.
    # Actually, vectorizing is faster.
    # Let's use a slightly smarter approach: 
    # Area of triangle (i,j,k) = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # We can iterate over pairs (i,j) and find the k that minimizes distance to line ij?
    # No, just brute force 286 is very fast.
    
    # Vectorized version for speed:
    # points: (N, 2)
    # We want min over i<j<k of 0.5 * |det(p_j - p_i, p_k - p_i)|
    
    # Let's stick to the loop for clarity and correctness, it's fast enough for N=13.
    # 286 iterations is trivial.
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        if len(points) < 3:
            return 0.0
        # Fallback
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5

def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial guess: Regular polygon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Strategy: Stochastic Local Search (Simulated Annealing)
    # The objective is non-smooth. Gradient-based methods get stuck.
    # We will perturb the point set and accept worse solutions probabilistically 
    # to escape local minima.
    
    rng = np.random.default_rng(42)
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Temperature schedule
    T = 0.01
    T_min = 1e-6
    alpha = 0.95
    
    max_iters = 50000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Perturbation: Move a random point by a small amount
        idx = rng.integers(0, N)
        # Step size should be adaptive? Start small.
        step_size = 0.01 * T + 0.001
        delta = rng.normal(0, step_size, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Calculate new score
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            # Always accept improvement
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept worse solution with probability exp(-delta_E / T)
            # delta_E is negative here (worse score)
            delta_E = current_score - new_score
            if delta_E < 0:
                delta_E = 0
            prob = np.exp(-delta_E / T)
            if rng.random() < prob:
                current_points = new_points
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_points
# EVOLVE-BLOCK-END
