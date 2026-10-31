"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the unit equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    if y < -eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > (3**0.5) * x + eps:
        return False
    if y > (3**0.5) * (1 - x) + eps:
        return False
    return True

def _clamp_to_triangle(p, margin=0.001):
    """Clamp a point to be inside the triangle with a small margin."""
    x, y = p
    x = max(margin, min(1 - margin, x))
    y = max(margin, min((3**0.5) * (x - margin), y))
    y = max(margin, min((3**0.5) * (1 - x + margin), y))
    return np.array([x, y])

def _coordinate_descent(pts, max_iters=200, step_sizes=None, time_limit=None, start_time=None):
    """Coordinate descent to maximize the minimum triangle area."""
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    n = len(pts)
    current_best = _min_triangle_area(pts)
    pts = pts.copy()
    
    for step_size in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_p = pts[i] + np.array([dx, dy])
                        if not _inside_triangle(new_p):
                            continue
                        old = pts[i]
                        pts[i] = new_p
                        area = _min_triangle_area(pts)
                        if area > current_best + 1e-10:
                            current_best = area
                            improved = True
                        else:
                            pts[i] = old
                if time_limit is not None and start_time is not None:
                    if time.time() - start_time > time_limit:
                        return pts, current_best
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
    
    return pts, current_best

def _generate_initial_configs(n_configs=20):
    """Generate diverse initial configurations."""
    configs = []
    rng = np.random.RandomState(42)
    
    # Config 1: Grid-like arrangement
    grid_pts = []
    # Place points on a coarse grid inside the triangle
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N
            y_max = (3**0.5) * min(x, 1-x)
            y = (j + 0.5) / N * y_max * 0.9
            if _inside_triangle([x, y]):
                grid_pts.append([x, y])
                if len(grid_pts) >= N:
                    break
        if len(grid_pts) >= N:
            break
    if len(grid_pts) == N:
        configs.append(np.array(grid_pts))
    
    # Config 2: Random with rejection
    for _ in range(10):
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                x = u + 0.5 * v
                y = (3**0.5) / 2 * v
                pts.append([x, y])
        configs.append(np.array(pts))
    
    # Config 3: Symmetric patterns - place some points at vertices and edges
    # Vertices
    v1 = [0.05, 0.05]
    v2 = [0.95, 0.05]
    v3 = [0.5, (3**0.5)/2 * 0.95]
    pts = [v1, v2, v3]
    # Add points along edges and interior
    for t in [0.25, 0.5, 0.75]:
        pts.append([t, 0.02])
    for t in [0.25, 0.5, 0.75]:
        pts.append([0.5 + t*0.48, (3**0.5)/2 * (1-t) * 0.95])
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.9:
            x = u + 0.5 * v
            y = (3**0.5) / 2 * v
            pts.append([x, y])
    configs.append(np.array(pts[:N]))
    
    # More random configs
    while len(configs) < n_configs:
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                x = u + 0.5 * v
                y = (3**0.5) / 2 * v
                pts.append([x, y])
        configs.append(np.array(pts))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate diverse initial configurations
    configs = _generate_initial_configs(25)
    
    best_pts = None
    best_area = 0.0
    
    for cfg_idx, pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Run coordinate descent with time limit
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            break
        
        # Use a portion of remaining time for this config
        time_limit = min(remaining_time * 0.4, 30)
        
        refined_pts, area = _coordinate_descent(
            pts, 
            max_iters=100, 
            step_sizes=[0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005],
            time_limit=time_limit,
            start_time=time.time()
        )
        
        if area > best_area:
            best_area = area
            best_pts = refined_pts.copy()
    
    # If no good result, return a reasonable default
    if best_pts is None:
        # Simple grid-like arrangement
        pts = []
        positions = [
            [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
            [0.25, 0.3], [0.5, 0.3], [0.75, 0.3],
            [0.15, 0.5], [0.5, 0.5], [0.85, 0.5],
            [0.3, 0.65], [0.7, 0.65]
        ]
        for p in positions:
            pts.append(_clamp_to_triangle(p))
        best_pts = np.array(pts)
    
    return best_pts

# EVOLVE-BLOCK-END
