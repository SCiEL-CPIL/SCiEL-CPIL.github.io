"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
DEADLINE = 140  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a / (S3H / 2)  # Normalize by triangle area (sqrt(3)/4)


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < 0 or y > S3H:
        return False
    if x < 0 or x > 1:
        return False
    # Check below the two slanted edges
    if y > S3H * (1 - x):  # right edge from (1,0) to (0.5, S3H)
        return False
    if y > S3H * x:  # left edge from (0,0) to (0.5, S3H)
        return False
    return True


def _penalized_min_area(x):
    """Objective for DE: minimize negative min area with penalty for points outside."""
    points = x.reshape(-1, 2)
    # Penalty for points outside triangle
    penalty = 0
    for i in range(N):
        if not _is_inside_triangle(points[i]):
            # Compute distance to triangle boundary
            x, y = points[i]
            d1 = max(0, y)  # bottom
            d2 = max(0, S3H - y)  # top
            d3 = max(0, x)  # left
            d4 = max(0, 1 - x)  # right
            d5 = max(0, S3H * (1 - x) - y)  # right slant
            d6 = max(0, S3H * x - y)  # left slant
            min_dist = min(d1, d2, d3, d4, d5, d6)
            penalty += 100 * min_dist ** 2
    min_a = _min_area_normalized(points)
    return -min_a + penalty


def _local_polish(points):
    """Nelder-Mead polish on the point configuration."""
    points = np.array(points, dtype=float)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        min_a = _min_area_normalized(pts)
        return -min_a
    
    x0 = points.flatten()
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    best_pts = res.x.reshape(-1, 2)
    
    # Ensure all points are inside triangle
    for i in range(N):
        if not _is_inside_triangle(best_pts[i]):
            # Project back to nearest interior point
            x, y = best_pts[i]
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, S3H * 0.99)
            # Further project to satisfy slant constraints
            if y > S3H * (1 - x):
                y = S3H * (1 - x) * 0.999
            if y > S3H * x:
                y = S3H * x * 0.999
            best_pts[i] = [x, y]
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Initial random configuration
    rng = np.random.RandomState(7)
    
    # Generate random points inside triangle using rejection sampling
    def random_point_in_triangle():
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                if 0.01 < x < 0.99 and 0.01 < y < S3H * 0.99:
                    return [x, y]
    
    # DE bounds: each coordinate in [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    best_points = None
    best_score = -float('inf')
    
    # Multiple restarts with DE
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Initial guess
        x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
        
        # Run differential evolution
        try:
            result = differential_evolution(
                _penalized_min_area, bounds, 
                seed=r, maxiter=100, tol=1e-8,
                popsize=15, mutation=(0.3, 1.2), recombination=0.7,
                polish=True
            )
        except Exception:
            continue
        
        if time.time() - start_time > DEADLINE - 10:
            break
        
        de_points = result.x.reshape(-1, 2)
        
        # Local polish
        polished = _local_polish(de_points)
        
        score = _min_area_normalized(polished)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Final high-precision polish on best result
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        best_points = _local_polish(best_points)
    
    # Verify all points are inside
    for i in range(N):
        if not _is_inside_triangle(best_points[i]):
            x, y = best_points[i]
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, S3H * 0.99)
            if y > S3H * (1 - x):
                y = S3H * (1 - x) * 0.999
            if y > S3H * x:
                y = S3H * x * 0.999
            best_points[i] = [x, y]
    
    return best_points
# EVOLVE-BLOCK-END
