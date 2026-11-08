import numpy as np
import time
import random
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

def _min_triangle_area(points):
    """Compute minimum area of all triangles formed by N points."""
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _project_to_triangle(p):
    """Project a point into the equilateral triangle (0,0), (1,0), (0.5, H)."""
    x, y = p
    # Barycentric coordinates for the triangle
    # Point is inside if all barycentric coords are in [0,1]
    # b0 = 1 - x - y/H
    # b1 = x
    # b2 = y/H
    b0 = 1.0 - x - y / H
    b1 = x
    b2 = y / H
    
    if b0 >= 0 and b1 >= 0 and b2 >= 0:
        return np.array([x, y])
    
    # Find closest point on boundary
    # Edge 1: from (0,0) to (1,0) - y = 0
    # Edge 2: from (1,0) to (0.5, H)
    # Edge 3: from (0.5, H) to (0,0)
    
    best = None
    best_dist = float('inf')
    
    # Edge 1: y = 0, x in [0, 1]
    x1 = np.clip(x, 0, 1)
    p1 = np.array([x1, 0.0])
    d1 = np.sum((p - p1)**2)
    if d1 < best_dist:
        best_dist = d1
        best = p1
    
    # Edge 2: from (1,0) to (0.5, H)
    # Line: x + y/H = 1, or H*x + y = H
    # Normal: (H, 1), unit normal: (H, 1)/sqrt(H^2+1)
    # Point on line closest to p:
    a, b, c = H, 1.0, -H  # H*x + y = H
    num = a * x + b * y + c
    den = a*a + b*b
    px = x - a * num / den
    py = y - b * num / den
    # Check if on segment
    t = (px - 1.0) / (0.5 - 1.0) if abs(0.5 - 1.0) > 1e-12 else 0
    if -1e-9 <= t <= 1 + 1e-9:
        d2 = np.sum((p - np.array([px, py]))**2)
        if d2 < best_dist:
            best_dist = d2
            best = np.array([px, py])
    else:
        # Check endpoints
        for ep in [np.array([1.0, 0.0]), np.array([0.5, H])]:
            d = np.sum((p - ep)**2)
            if d < best_dist:
                best_dist = d
                best = ep
    
    # Edge 3: from (0.5, H) to (0,0)
    # Line: y = H*2*x, or 2H*x - y = 0
    a, b, c = 2*H, -1.0, 0
    num = a * x + b * y + c
    den = a*a + b*b
    px = x - a * num / den
    py = y - b * num / den
    t = px / 0.5 if abs(0.5) > 1e-12 else 0
    if -1e-9 <= t <= 1 + 1e-9:
        d3 = np.sum((p - np.array([px, py]))**2)
        if d3 < best_dist:
            best_dist = d3
            best = np.array([px, py])
    else:
        for ep in [np.array([0.5, H]), np.array([0.0, 0.0])]:
            d = np.sum((p - ep)**2)
            if d < best_dist:
                best_dist = d
                best = ep
    
    return best

def _objective_flat(x):
    """Objective for optimization: negative min triangle area."""
    points = x.reshape(N, 2)
    return -_min_triangle_area(points)

def _is_valid(points):
    """Check if all points are inside the triangle."""
    for p in points:
        x, y = p
        b0 = 1.0 - x - y / H
        b1 = x
        b2 = y / H
        if b0 < -1e-10 or b1 < -1e-10 or b2 < -1e-10:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    rng = random.Random(42)
    
    # Initial points: try a few different starting configurations
    best_points = None
    best_area = -1.0
    
    # Strategy 1: Start with vertices and centroid, then add points
    base_points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H/3],  # centroid
    ]
    
    # Add more points in a grid-like pattern
    for i in range(7):
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95)
        if u + v < 1.0:
            pt = [u + 0.5*v, H*v]
            base_points.append(pt)
    
    points = np.array(base_points[:N])
    
    # Local optimization with Nelder-Mead
    x0 = points.flatten()
    
    def objective_with_penalty(x):
        pts = x.reshape(N, 2)
        # Check if valid
        for p in pts:
            x_, y_ = p
            b0 = 1.0 - x_ - y_ / H
            b1 = x_
            b2 = y_ / H
            if b0 < -0.05 or b1 < -0.05 or b2 < -0.05:
                return 1000.0  # Large penalty
        return -_min_triangle_area(pts)
    
    result = minimize(objective_with_penalty, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    if result.success or result.fun < 0:
        pts = result.x.reshape(N, 2)
        # Project to valid region
        valid_pts = []
        for p in pts:
            valid_pts.append(_project_to_triangle(p))
        pts = np.array(valid_pts)
        if _is_valid(pts):
            area = _min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts.copy()
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initial points with some structure
        pts = []
        # Include vertices for better coverage
        pts.append([0.0, 0.0])
        pts.append([1.0, 0.0])
        pts.append([0.5, H])
        
        # Add centroid
        pts.append([0.5, H/3])
        
        # Add remaining points randomly
        while len(pts) < N:
            u = rng.uniform(0.02, 0.98)
            v = rng.uniform(0.02, 0.98)
            if u + v < 0.98:
                pt = [u + 0.5*v, H*v]
                pts.append(pt)
        
        pts = np.array(pts)
        
        x0 = pts.flatten()
        res = minimize(objective_with_penalty, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < 0:
            pts_opt = res.x.reshape(N, 2)
            valid_pts = []
            for p in pts_opt:
                valid_pts.append(_project_to_triangle(p))
            pts_opt = np.array(valid_pts)
            if _is_valid(pts_opt):
                area = _min_triangle_area(pts_opt)
                if area > best_area:
                    best_area = area
                    best_points = pts_opt.copy()
    
    # Final polish with more iterations on the best
    if best_points is not None and time.time() < deadline:
        x0 = best_points.flatten()
        res = minimize(objective_with_penalty, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < 0:
            pts_opt = res.x.reshape(N, 2)
            valid_pts = []
            for p in pts_opt:
                valid_pts.append(_project_to_triangle(p))
            pts_opt = np.array(valid_pts)
            if _is_valid(pts_opt):
                area = _min_triangle_area(pts_opt)
                if area > best_area:
                    best_area = area
                    best_points = pts_opt.copy()
    
    if best_points is None:
        # Fallback: return the initial heuristic points
        best_points = np.array(base_points[:N])
    
    return best_points
