"""Heilbronn triangles: maximize the smallest triangle area among 11 points in a unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0  # area of the equilateral triangle with side 1


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the given points."""
    n = len(pts)
    min_a = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a


def _clip_to_triangle(pts):
    """Clip points to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    pts = pts.copy()
    # The three half-planes:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (below the left edge from (0,0) to (0.5, sqrt(3)/2))
    # 3. y <= sqrt(3) * (1 - x)  (below the right edge from (1,0) to (0.5, sqrt(3)/2))
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0.0
        # Left edge: line from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3) * x
        if y > SQRT3 * x:
            # Project onto the line
            # The line: sqrt(3)*x - y = 0
            # Distance-based projection
            t = x + y / SQRT3
            t = t / (1 + 1 / 3)  # = 3t/4
            x = 0.5 * t
            y = SQRT3 * x
        # Right edge: line from (1,0) to (0.5, sqrt(3)/2): y = sqrt(3)*(1-x)
        if y > SQRT3 * (1 - x):
            t = x + y / SQRT3
            t = t / (1 + 1 / 3)
            x = 1 - 0.5 * t
            y = SQRT3 * (1 - x)
        pts[i] = [x, y]
    return pts


def _objective(x_flat):
    """Objective to minimize: negative of the minimum triangle area (with penalty for out-of-bounds)."""
    pts = x_flat.reshape(N, 2)
    # Check if all points are inside the triangle (with small tolerance)
    eps = 1e-6
    for i in range(N):
        x, y = pts[i]
        if y < -eps or y > SQRT3 * x + eps or y > SQRT3 * (1 - x) + eps:
            # Heavy penalty for being outside
            return 1.0  # large positive value to be minimized
    return -_min_triangle_area(pts)


def _sobol_sequence(n, seed=0):
    """Generate n points in [0,1]^2 using a simple low-discrepancy approach (Halton-like)."""
    points = []
    base1 = 2
    base2 = 3
    for i in range(1, n + 1):
        # Halton sequence in base 2
        x1 = 0.0
        f = 1.0 / base1
        i1 = i
        while i1 > 0:
            x1 += f * (i1 % base1)
            i1 //= base1
            f /= base1
        # Halton sequence in base 3
        x2 = 0.0
        f = 1.0 / base2
        i2 = i
        while i2 > 0:
            x2 += f * (i2 % base2)
            i2 //= base2
            f /= base2
        points.append([x1, x2])
    return np.array(points)


def _initial_points_sobol(seed_offset=0):
    """Generate initial 11 points inside the triangle using a Sobol/Halton-like sequence."""
    raw = _sobol_sequence(N, seed=seed_offset)
    # Map from [0,1]^2 to the triangle
    pts = np.zeros((N, 2))
    for i in range(N):
        u, v = raw[i]
        # Ensure u+v < 1
        if u + v >= 1:
            s = 1.0 / (u + v)
            u, v = u * s, v * s
        pts[i] = [u + 0.5 * v, SQRT3 / 2 * v]
    return pts


def _nelder_mead_optimize(pts0, time_limit=10.0):
    """Run Nelder-Mead optimization starting from pts0, with a time limit."""
    start_time = time.time()
    x0 = pts0.flatten()
    
    # Define objective with time check
    best_x = x0.copy()
    best_val = _objective(x0)
    
    def obj_with_time(x):
        nonlocal best_x, best_val
        if time.time() - start_time > time_limit:
            return best_val  # Stop optimizing
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        return val
    
    try:
        res = minimize(obj_with_time, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    except Exception:
        pass
    
    if time.time() - start_time <= time_limit:
        return best_x.reshape(N, 2)
    return best_x.reshape(N, 2)


def construct():
    deadline = 110.0  # Leave 10s safety margin before 120s limit
    start_time = time.time()
    
    best_pts = None
    best_score = -np.inf
    
    # Multi-start with different Sobol offsets
    num_starts = 20
    for s in range(num_starts):
        if time.time() - start_time > deadline:
            break
        
        pts0 = _initial_points_sobol(seed_offset=s * 7 + 1)
        pts0 = _clip_to_triangle(pts0)
        
        # Quick Nelder-Mead
        time_remaining = deadline - (time.time() - start_time)
        if time_remaining < 2:
            break
        
        tm = min(8.0, time_remaining * 0.8)
        pts_opt = _nelder_mead_optimize(pts0, time_limit=tm)
        pts_opt = _clip_to_triangle(pts_opt)
        
        score = _min_triangle_area(pts_opt)
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # If we have time, do a few more refined starts around the best solution
    if best_pts is not None and time.time() - start_time < deadline - 10:
        for s in range(5):
            if time.time() - start_time > deadline:
                break
            # Perturb the best solution
            noise = np.random.randn(N, 2) * 0.05
            pts_pert = best_pts + noise
            pts_pert = _clip_to_triangle(pts_pert)
            
            time_remaining = deadline - (time.time() - start_time)
            if time_remaining < 2:
                break
            tm = min(5.0, time_remaining * 0.8)
            pts_opt = _nelder_mead_optimize(pts_pert, time_limit=tm)
            pts_opt = _clip_to_triangle(pts_opt)
            
            score = _min_triangle_area(pts_opt)
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    if best_pts is None:
        # Fallback: return the seed-like random points
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, SQRT3 / 2 * v])
        return np.array(points)
    
    return best_pts
