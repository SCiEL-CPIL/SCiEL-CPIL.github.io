"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum area among all C(n,3) triangles."""
    n = len(points)
    min_area = float('inf')
    min_triple = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
                    min_triple = (i, j, k)
    return min_area, min_triple

def _inside_triangle(p, tol=1e-9):
    """Check if point is inside or on boundary of equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    h = np.sqrt(3) / 2
    x, y = p
    if y < -tol:
        return False
    if y > h + tol:
        return False
    # Check left edge: from (0,0) to (0.5, h): y <= h*x => x >= y/h
    # Check right edge: from (1,0) to (0.5, h): y <= h*(1-x) => x <= 1 - y/h
    if x < -tol or x > 1 + tol:
        return False
    # Left boundary: line from (0,0) to (0.5, h): slope = h/0.5 = 2h = sqrt(3)
    # So y <= sqrt(3)*x => x >= y/sqrt(3)
    if x < y / np.sqrt(3) - tol:
        return False
    # Right boundary: line from (1,0) to (0.5, h): slope = h/(0.5-1) = -sqrt(3)
    # y <= -sqrt(3)*(x-1) => x <= 1 - y/sqrt(3)
    if x > 1 - y / np.sqrt(3) + tol:
        return False
    return True

def _penalized_objective(x_flat, points, min_area_target, penalty_weight=100.0):
    """Objective: minimize -min_area + penalty for points outside triangle."""
    pts = x_flat.reshape(-1, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for p in pts:
        if not _inside_triangle(p, tol=1e-6):
            # Distance from triangle
            h = np.sqrt(3) / 2
            x, y = p
            # Distance to each edge
            # Edge 1: y=0, distance = max(0, -y)
            d1 = max(0, -y)
            # Edge 2: y = sqrt(3)*x, distance = max(0, sqrt(3)*x - y) / 2
            d2 = max(0, np.sqrt(3) * x - y) / 2
            # Edge 3: y = -sqrt(3)*(x-1), distance = max(0, -sqrt(3)*(x-1) - y) / 2
            d3 = max(0, -np.sqrt(3) * (x - 1) - y) / 2
            penalty += max(d1, d2, d3)
    ma, _ = _min_triangle_area(pts)
    return -ma + penalty_weight * penalty

def _local_optimize(points, max_iter=50, tol=1e-8):
    """Iteratively improve by moving points to increase min triangle area."""
    points = points.copy()
    for _ in range(max_iter):
        ma, triple = _min_triangle_area(points)
        if triple is None:
            break
        i, j, k = triple
        # Try to move each of the three points to improve
        best_improvement = 0
        best_move = None
        for idx in [i, j, k]:
            # Try small perturbations in random directions
            for _ in range(20):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                for step in [0.01, 0.005, 0.001]:
                    new_pts = points.copy()
                    new_pts[idx] = points[idx] + step * direction
                    if all(_inside_triangle(p, tol=1e-6) for p in new_pts):
                        new_ma, _ = _min_triangle_area(new_pts)
                        improvement = new_ma - ma
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (idx, step, direction)
        if best_move is not None:
            idx, step, direction = best_move
            points[idx] = points[idx] + step * direction
        else:
            break
    return points

def _scipy_optimize(points, max_time=5.0):
    """Use scipy to optimize points to maximize min triangle area."""
    start_time = time.time()
    x0 = points.flatten()
    
    def objective(x_flat):
        pts = x_flat.reshape(-1, 2)
        ma, _ = _min_triangle_area(pts)
        # Add penalty for points outside triangle
        penalty = 0.0
        for p in pts:
            if not _inside_triangle(p, tol=1e-6):
                h = np.sqrt(3) / 2
                x, y = p
                d1 = max(0, -y)
                d2 = max(0, np.sqrt(3) * x - y) / 2
                d3 = max(0, -np.sqrt(3) * (x - 1) - y) / 2
                penalty += max(d1, d2, d3)
        return -ma + 100.0 * penalty
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.success or (time.time() - start_time) < max_time:
            new_pts = result.x.reshape(-1, 2)
            if all(_inside_triangle(p, tol=1e-6) for p in new_pts):
                return new_pts
    except:
        pass
    return points

def _generate_initial_config(seed=0):
    """Generate a structured initial configuration."""
    rng = np.random.RandomState(seed)
    configs = []
    
    # Config 1: Regular grid-like placement
    # Place points in a triangular lattice pattern
    h = np.sqrt(3) / 2
    # 4-3-2-2 pattern or similar
    rows = [4, 3, 2, 2]  # 4+3+2+2 = 11
    pts = []
    total_rows = len(rows)
    for row_idx, count in enumerate(rows):
        y = h * (row_idx + 0.5) / total_rows
        x_start = 0.5 * y / h  # Left boundary at this height
        x_end = 1 - 0.5 * y / h  # Right boundary at this height
        if count == 1:
            pts.append([(x_start + x_end) / 2, y])
        else:
            for i in range(count):
                x = x_start + (x_end - x_start) * (i + 0.5) / count
                pts.append([x, y])
    configs.append(np.array(pts))
    
    # Config 2: Symmetric with center
    pts2 = []
    # Center
    pts2.append([0.5, h / 3])
    # 3 points near corners
    eps = 0.1
    pts2.append([eps, eps * np.sqrt(3)])
    pts2.append([1 - eps, eps * np.sqrt(3)])
    pts2.append([0.5, h - eps * np.sqrt(3)])
    # 6 more points in between
    for angle in range(6):
        theta = np.pi / 3 * angle + np.pi / 6
        r = 0.3
        x = 0.5 + r * np.cos(theta)
        y = h / 3 + r * np.sin(theta)
        if _inside_triangle([x, y], tol=1e-6):
            pts2.append([x, y])
        else:
            # Project to triangle
            pts2.append([0.5, h / 3])
    # Ensure we have 11
    while len(pts2) < 11:
        pts2.append([0.5, h / 3])
    configs.append(np.array(pts2[:11]))
    
    # Config 3: Random with rejection
    pts3 = []
    for _ in range(100):
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = h * v
            pts3.append([x, y])
            if len(pts3) == 11:
                break
    configs.append(np.array(pts3))
    
    return configs

def construct():
    """Place 11 points to maximize minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    best_points = None
    best_area = 0.0
    
    # Try multiple initial configurations
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        
        configs = _generate_initial_config(seed)
        for config in configs:
            if time.time() > deadline - 5:
                break
            
            if len(config) != N:
                continue
            
            # Validate initial config
            if not all(_inside_triangle(p, tol=1e-6) for p in config):
                continue
            
            ma, _ = _min_triangle_area(config)
            if ma > best_area:
                best_area = ma
                best_points = config.copy()
            
            # Local optimization
            if time.time() > deadline - 5:
                break
            optimized = _local_optimize(config, max_iter=30)
            ma_opt, _ = _min_triangle_area(optimized)
            if ma_opt > best_area:
                best_area = ma_opt
                best_points = optimized.copy()
            
            # Scipy optimization
            if time.time() > deadline - 5:
                break
            scipy_result = _scipy_optimize(optimized, max_time=3.0)
            ma_scipy, _ = _min_triangle_area(scipy_result)
            if ma_scipy > best_area:
                best_area = ma_scipy
                best_points = scipy_result.copy()
    
    # Final validation
    if best_points is None:
        # Fallback to seed-like random
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
    
    return best_points
# EVOLVE-BLOCK-END
