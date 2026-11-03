import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11

def point_in_triangle(p, a, b, c):
    # p: (2,) or (N, 2)
    # a, b, c: (2,)
    p = np.atleast_2d(p)
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    
    v0 = c[0] - a[0]
    v1 = b[0] - a[0]
    v2 = p - a[0]
    
    d00 = np.sum(v0 * v0)
    d01 = np.sum(v0 * v1)
    d02 = np.sum(v0 * v2, axis=1)
    d11 = np.sum(v1 * v1)
    d12 = np.sum(v1 * v2, axis=1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.zeros(len(p), dtype=bool)
        
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    
    return (u >= 0) & (v >= 0) & (u + v <= 1)

def min_triangle_area(points):
    n = len(points)
    if not hasattr(min_triangle_area, "indices"):
        idx = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    idx.append((i, j, k))
        min_triangle_area.indices = np.array(idx)
        
    indices = min_triangle_area.indices
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def construct():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    best_points = None
    best_score = -1.0
    
    # Precompute triangle indices
    indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                indices.append((i, j, k))
    indices = np.array(indices)
    
    def calc_areas(points):
        p1 = points[indices[:, 0]]
        p2 = points[indices[:, 1]]
        p3 = points[indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    # Soft-min objective for smooth optimization
    def objective_soft(x, t=100.0):
        points = x.reshape(-1, 2)
        areas = calc_areas(points)
        # log-sum-exp trick for numerical stability
        # min(areas) approx - (1/t) * log(sum(exp(-t * areas)))
        # We want to maximize min(areas), so minimize -min(areas).
        # -min(areas) approx (1/t) * log(sum(exp(-t * areas)))
        max_a = np.max(areas)
        if max_a == 0:
            return 1e6
        # Shift to prevent underflow
        shifted = -t * (areas - max_a) 
        # Wait, if areas are small, -t*areas is large negative.
        # We want sum(exp(-t * areas)).
        # Let m = min(areas). exp(-t * areas) = exp(-t * m) * exp(-t * (areas - m)).
        # log(sum(exp(-t * areas))) = -t * m + log(sum(exp(-t * (areas - m))))
        # We want to minimize this value? No.
        # Objective: minimize -min(areas).
        # -min(areas) = (1/t) * log(sum(exp(-t * areas)))
        # So we minimize (1/t) * log(sum(exp(-t * areas)))
        
        # Numerical stability:
        # areas are positive.
        # Let A = areas.
        # val = (1/t) * log(sum(exp(-t * A)))
        # log_sum_exp(-t * A)
        # max_val = np.max(-t * A) = -t * np.min(A)
        # sum(exp(-t * A)) = exp(max_val) * sum(exp(-t * A - max_val))
        # log(sum(exp(-t * A))) = max_val + log(sum(exp(-t * A - max_val)))
        # -t * A - max_val = -t * A - (-t * min_A) = -t * (A - min_A) <= 0.
        
        min_a = np.min(areas)
        if min_a < 1e-9:
             return 1e6
        exponent = -t * (areas - min_a)
        log_sum = np.log(np.sum(np.exp(exponent)))
        val = (1.0 / t) * (-t * min_a + log_sum)
        return val

    # Constraints for SLSQP
    # y >= 0
    # y <= sqrt(3) x
    # y <= sqrt(3) (1 - x)
    # x >= 0, x <= 1
    
    def constraint_y_min(x):
        points = x.reshape(-1, 2)
        return points[:, 1]
        
    def constraint_y_max_left(x):
        points = x.reshape(-1, 2)
        return np.sqrt(3) * points[:, 0] - points[:, 1]
        
    def constraint_y_max_right(x):
        points = x.reshape(-1, 2)
        return np.sqrt(3) * (1.0 - points[:, 0]) - points[:, 1]
        
    def constraint_x_min(x):
        points = x.reshape(-1, 2)
        return points[:, 0]
        
    def constraint_x_max(x):
        points = x.reshape(-1, 2)
        return 1.0 - points[:, 0]

    constraints = [
        {'type': 'ineq', 'fun': constraint_y_min},
        {'type': 'ineq', 'fun': constraint_y_max_left},
        {'type': 'ineq', 'fun': constraint_y_max_right},
        {'type': 'ineq', 'fun': constraint_x_min},
        {'type': 'ineq', 'fun': constraint_x_max}
    ]
    
    # Initial guesses
    initial_guesses = []
    
    # Guess 1: Triangular Lattice
    # 4 points on bottom, 3 on next, 2 on next, 1 on top? Total 10.
    # We need 11.
    # Let's try a 4-4-3 split.
    # Or 5-3-3?
    # Standard hexagonal packing in triangle.
    # Let's use a simple grid and project.
    
    # Generate points on a triangular grid
    pts = []
    # 4 rows
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point
    # Total 10. Add one more?
    # Maybe 4-4-3?
    # Let's just use a dense random search to find a good start, then refine.
    
    rng = np.random.RandomState(123)
    
    # Generate 50 random valid configurations
    for _ in range(50):
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * b + v * c
                pts.append(p)
        # Ensure no duplicates (unlikely with continuous)
        initial_guesses.append(np.array(pts).flatten())
        
    # Add a structured guess:
    # 4 points at y=0.1, 4 at y=0.3, 3 at y=0.5
    # x positions adjusted to fit in triangle width at that y
    h = np.sqrt(3)/2
    def width_at_y(y):
        # Triangle sides: y = sqrt(3)x and y = -sqrt(3)(x-1)
        # x_left = y / sqrt(3)
        # x_right = 1 - y / sqrt(3)
        # width = x_right - x_left = 1 - 2y/sqrt(3)
        return 1.0 - 2.0 * y / np.sqrt(3)
        
    pts_struct = []
    # Row 1: y=0.15, 4 points
    y1 = 0.15
    w1 = width_at_y(y1)
    x_start1 = y1 / np.sqrt(3)
    for i in range(4):
        x = x_start1 + (i + 0.5) * (w1 / 4.0)
        pts_struct.append([x, y1])
        
    # Row 2: y=0.35, 4 points
    y2 = 0.35
    w2 = width_at_y(y2)
    x_start2 = y2 / np.sqrt(3)
    for i in range(4):
        x = x_start2 + (i + 0.5) * (w2 / 4.0)
        pts_struct.append([x, y2])
        
    # Row 3: y=0.55, 3 points
    y3 = 0.55
    w3 = width_at_y(y3)
    x_start3 = y3 / np.sqrt(3)
    for i in range(3):
        x = x_start3 + (i + 0.5) * (w3 / 3.0)
        pts_struct.append([x, y3])
        
    initial_guesses.append(np.array(pts_struct).flatten())
    
    # Optimize each guess
    t_values = [50.0, 100.0, 200.0]
    
    for x0 in initial_guesses:
        # Check if initial guess is valid
        points0 = x0.reshape(-1, 2)
        if not np.all(point_in_triangle(points0, a, b, c)):
            continue
            
        # Annealing on t
        current_x = x0
        for t in t_values:
            try:
                res = minimize(objective_soft, current_x, method='SLSQP', 
                               bounds=[(0, 1)] * 22, # Loose bounds, constraints handle the rest
                               constraints=constraints,
                               options={'maxiter': 200, 'ftol': 1e-12})
                if res.success or res.fun < 1e5:
                    current_x = res.x
            except Exception:
                break
                
        # Final evaluation with exact min
        final_points = current_x.reshape(-1, 2)
        if np.all(point_in_triangle(final_points, a, b, c)):
            final_area = min_triangle_area(final_points)
            if final_area > best_score:
                best_score = final_area
                best_points = final_points
                # Add jittered version for further exploration
                if len(initial_guesses) < 100:
                    jitter = np.random.normal(0, 0.005, size=current_x.shape)
                    new_x0 = current_x + jitter
                    # Project back to valid? SLSQP will handle it.
                    initial_guesses.append(new_x0)
                    
    if best_points is None:
        # Fallback to the structured guess
        best_points = np.array(pts_struct)
        
    return best_points
