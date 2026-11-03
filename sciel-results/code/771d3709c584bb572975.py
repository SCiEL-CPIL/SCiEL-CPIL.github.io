"""Heilbronn triangles: place 11 points to maximize minimum triangle area."""
import time
import numpy as np
from itertools import combinations

N = 11
TRI_AREA = (3 ** 0.5) / 4.0  # area of unit equilateral triangle

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs(
                    (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                    (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1])
                )
                if a < min_a:
                    min_a = a
    return min_a

def _inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9:
        return False
    # Line from (0,0) to (1,0): y >= 0 (checked)
    # Line from (1,0) to (0.5, sqrt(3)/2): y <= sqrt(3)*(1-x)
    if y > (3 ** 0.5) * (1 - x) + 1e-9:
        return False
    # Line from (0.5, sqrt(3)/2) to (0,0): y <= sqrt(3)*x
    if y > (3 ** 0.5) * x + 1e-9:
        return False
    return True

def _clip_to_triangle(pts):
    """Clip points to be inside the triangle."""
    result = []
    for p in pts:
        x, y = p
        # Clip to triangle boundaries
        # Boundary 1: y >= 0
        y = max(0.0, y)
        # Boundary 2: y <= sqrt(3)*(1-x)
        if y > (3 ** 0.5) * (1 - x):
            # Project onto this line
            # Line: y = sqrt(3)*(1-x) => sqrt(3)*x + y - sqrt(3) = 0
            a, b, c = 3 ** 0.5, 1.0, -3 ** 0.5
            d = (a * x + b * y + c) / (a * a + b * b)
            x = x - d * a
            y = y - d * b
        # Boundary 3: y <= sqrt(3)*x
        if y > (3 ** 0.5) * x:
            # Line: y = sqrt(3)*x => -sqrt(3)*x + y = 0
            a, b, c = -3 ** 0.5, 1.0, 0.0
            d = (a * x + b * y + c) / (a * a + b * b)
            x = x - d * a
            y = y - d * b
        # Final safety clip
        x = max(0.0, min(1.0, x))
        y = max(0.0, min((3 ** 0.5) / 2, y))
        if y > (3 ** 0.5) * (1 - x):
            y = (3 ** 0.5) * (1 - x)
        if y > (3 ** 0.5) * x:
            y = (3 ** 0.5) * x
        result.append([x, y])
    return np.array(result)

def _initial_grid():
    """Create an initial structured configuration."""
    # Use a triangular lattice-like arrangement
    # 3-3-3-2 pattern or similar
    pts = []
    # Bottom row (y near 0)
    for i in range(4):
        x = 0.15 + i * 0.2
        y = 0.05
        pts.append([x, y])
    # Middle row
    for i in range(4):
        x = 0.2 + i * 0.15
        y = 0.25
        pts.append([x, y])
    # Top row
    for i in range(3):
        x = 0.35 + i * 0.15
        y = 0.45
        pts.append([x, y])
    return _clip_to_triangle(np.array(pts))

def _initial_random(seed):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            x = u + 0.5 * v
            y = (3 ** 0.5) / 2 * v
            pts.append([x, y])
    return _clip_to_triangle(np.array(pts))

def _local_optimize(pts, max_iter=200, step_size=0.02):
    """Simple local optimization: try small perturbations and keep improvements."""
    pts = pts.copy()
    current_val = _min_triangle_area(pts)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [pts[i, 0] + dx, pts[i, 1] + dy]
                    new_pts = _clip_to_triangle(new_pts)
                    new_val = _min_triangle_area(new_pts)
                    if new_val > current_val + 1e-10:
                        pts = new_pts
                        current_val = new_val
                        improved = True
        if not improved:
            break
    return pts, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s
    
    best_pts = None
    best_val = 0.0
    
    # Try multiple initial configurations
    initial_configs = []
    initial_configs.append(_initial_grid())
    for seed in range(20):
        initial_configs.append(_initial_random(seed))
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Local optimization with adaptive step size
        pts = init_pts.copy()
        val = _min_triangle_area(pts)
        
        if val > best_val:
            best_pts = pts.copy()
            best_val = val
        
        # Multi-scale local optimization
        for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() > deadline:
                break
            max_iter = max(50, int((deadline - time.time()) * 10))
            pts, val = _local_optimize(pts, max_iter=max_iter, step_size=step)
            if val > best_val:
                best_pts = pts.copy()
                best_val = val
    
    # Final check
    if best_pts is None:
        best_pts = _initial_grid()
    
    # Ensure valid
    best_pts = _clip_to_triangle(best_pts)
    return best_pts
