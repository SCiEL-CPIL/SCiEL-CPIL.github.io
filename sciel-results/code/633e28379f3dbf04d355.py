import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle index arrays for speed
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    p_i = points[_idx_i]
    p_j = points[_idx_j]
    p_k = points[_idx_k]
    
    dx_j = p_j[:, 0] - p_i[:, 0]
    dy_j = p_j[:, 1] - p_i[:, 1]
    dx_k = p_k[:, 0] - p_i[:, 0]
    dy_k = p_k[:, 1] - p_i[:, 1]
    
    areas = 0.5 * np.abs(dx_j * dy_k - dy_j * dx_k)
    return np.min(areas)

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y/SQRT3 - eps) and (x <= 1 - y/SQRT3 + eps)

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def project_points_to_triangle(points):
    """Project all points onto the triangle."""
    for i in range(len(points)):
        points[i] = project_to_triangle(points[i])
    return points

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points to make 11
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def max_min_distance(points):
    """Compute the maximum minimum pairwise distance (surrogate for spread)."""
    n = len(points)
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    return min_dist

def generate_maxmin_initial():
    """Generate initial points by maximizing minimum pairwise distance (Lloyd's algorithm style)."""
    rng = np.random.RandomState(42)
    # Start with structured points, then refine with Lloyd-like iterations
    points = generate_structured_initial().copy()
    
    for _ in range(100):
        # For each point, try to move it to increase min distance
        for i in range(N):
            best_p = points[i].copy()
            best_d = max_min_distance(points)
            for _ in range(30):
                dx = rng.uniform(-0.1, 0.1, 2)
                new_p = project_to_triangle(points[i] + dx)
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                old_p = points[i]
                points[i] = new_p
                d = max_min_distance(points)
                if d > best_d:
                    best_d = d
                    best_p = new_p
                else:
                    points[i] = old_p
            points[i] = best_p
    
    return points

def generate_random_points(seed):
    """Generate random points inside the triangle using rejection sampling."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1:
            # Barycentric coordinates
            a = 1 - u - v
            b = u
            c = v
            x = b * 1.0 + c * 0.5
            y = c * H
            points.append([x, y])
    return np.array(points)

def nelder_mead_optimize(points, deadline=None, max_iter=200):
    """Optimize using Nelder-Mead on the flattened point coordinates."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        # Penalize points outside the triangle
        for i in range(N):
            if not is_inside_triangle(pts[i], eps=1e-6):
                return -1e10
        return -min_triangle_area(pts)
    
    x0 = points.flatten()
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        opt_points = result.x.reshape(N, 2)
        # Project back into triangle
        opt_points = project_points_to_triangle(opt_points)
        opt_val = min_triangle_area(opt_points)
        return opt_points, opt_val
    except Exception:
        return points.copy(), min_triangle_area(points)

def local_optimize(points, max_iter=500, step_size=0.1, deadline=None):
    """Local optimization using coordinate descent / hill climbing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    rng = np.random.RandomState(123)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            best_local = points[i].copy()
            best_local_val = best_val
            
            for _ in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                    
                dx = rng.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_points = None
    best_value = -1
    
    # Strategy 1: Max-min distance initialization + Nelder-Mead
    if time.time() < deadline:
        init1 = generate_maxmin_initial()
        opt1, val1 = nelder_mead_optimize(init1, deadline=deadline, max_iter=200)
        if val1 > best_value:
            best_value = val1
            best_points = opt1.copy()
    
    # Strategy 2: Structured initialization + Nelder-Mead
    if time.time() < deadline:
        init2 = generate_structured_initial()
        opt2, val2 = nelder_mead_optimize(init2, deadline=deadline, max_iter=200)
        if val2 > best_value:
            best_value = val2
            best_points = opt2.copy()
    
    # Strategy 3: Random initializations + Nelder-Mead
    for seed in range(8):
        if time.time() > deadline - 15:
            break
        init_r = generate_random_points(seed * 42 + 7)
        opt_r, val_r = nelder_mead_optimize(init_r, deadline=deadline, max_iter=100)
        if val_r > best_value:
            best_value = val_r
            best_points = opt_r.copy()
    
    # Final local optimization polish on the best found
    if best_points is not None and time.time() < deadline - 5:
        final_points, final_val = local_optimize(best_points, max_iter=200, step_size=0.05, deadline=deadline)
        if final_val > best_value:
            best_value = final_val
            best_points = final_points.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i in range(len(best_points)):
            if not is_inside_triangle(best_points[i], eps=1e-6):
                best_points[i] = project_to_triangle(best_points[i])
    
    if best_points is None:
        best_points = generate_structured_initial()
    
    return best_points
