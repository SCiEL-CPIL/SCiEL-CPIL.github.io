"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _bary_to_cart(b):
    """Convert barycentric coords (u, v) with u,v>=0, u+v<=1 to cartesian in unit triangle."""
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # Point = u*A + v*B + w*C where w = 1-u-v
    # Actually standard: P = u*A + v*B + w*C
    # Let's use: P = (1-u-v)*A + u*B + v*C
    # A=(0,0), B=(1,0), C=(0.5, h)
    x = b[0] * 1.0 + b[1] * 0.5
    y = b[1] * (3 ** 0.5) / 2
    return x, y

def _cart_to_bary(x, y):
    """Convert cartesian to barycentric (u, v) where P = u*B + v*C + (1-u-v)*A."""
    h = (3 ** 0.5) / 2
    v = y / h
    u = x - 0.5 * v
    return u, v

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _negative_min_area_bary(bary_coords):
    """Objective: negative of min triangle area, given barycentric coords."""
    # Convert to cartesian
    h = (3 ** 0.5) / 2
    xs = bary_coords[0::2] + 0.5 * bary_coords[1::2]
    ys = bary_coords[1::2] * h
    points = np.column_stack([xs, ys])
    return -_min_triangle_area(points)

def _is_valid_bary(bary_coords):
    """Check if barycentric coordinates are valid (inside or on triangle)."""
    u = bary_coords[0::2]
    v = bary_coords[1::2]
    w = 1.0 - u - v
    return np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(w >= -1e-9)

def _clamp_to_triangle(bary_coords):
    """Clamp barycentric coordinates to be inside the triangle."""
    u = bary_coords[0::2].copy()
    v = bary_coords[1::2].copy()
    # Clamp u, v >= 0 and u + v <= 1
    u = np.maximum(u, 0.0)
    v = np.maximum(v, 0.0)
    s = u + v
    if s > 1.0:
        scale = 1.0 / s
        u *= scale
        v *= scale
    result = np.empty(2 * N)
    result[0::2] = u
    result[1::2] = v
    return result

def construct():
    start_time = time.time()
    
    # Initial best: use a symmetric arrangement as starting point
    # For 11 points, try a hexagonal-like packing
    best_bary = None
    best_score = -float('inf')
    
    # Seed 1: symmetric pattern
    # Center + ring of 6 + ring of 4? Let's try center, 6 around, 4 more
    h = (3 ** 0.5) / 2
    
    # Try a few good initial configurations
    initial_configs = []
    
    # Config 1: 3x3 grid-like in barycentric
    # Points at various barycentric positions
    grid_bary = []
    # Center
    grid_bary.append([0.5, 0.5])  # This is actually outside... u+v=1.0, ok on edge? No, w=0
    # Let me think more carefully. Barycentric: u+v+w=1, all >= 0
    # Center of triangle: u=v=w=1/3
    grid_bary = [[1/3, 1/3]]
    # 6 points around center at radius ~0.3
    for angle in range(0, 360, 60):
        a = np.radians(angle)
        # In barycentric, moving from center
        # This is tricky. Let me just use cartesian and convert.
    
    # Let's work in cartesian for initial configs, then convert
    def cart_config_to_bary(config):
        bary = []
        for x, y in config:
            u, v = _cart_to_bary(x, y)
            bary.append([u, v])
        return bary
    
    # Config 1: symmetric hexagonal packing
    # 1 center, 6 in first ring, 4 in second ring (partial)
    config1 = [(0.5, h/3)]  # center
    r1 = 0.25
    for angle in range(0, 360, 60):
        a = np.radians(angle)
        config1.append((0.5 + r1 * np.cos(a), h/3 + r1 * np.sin(a)))
    # That's 7 points. Need 4 more.
    r2 = 0.45
    for angle in range(30, 360, 60):
        a = np.radians(angle)
        config1.append((0.5 + r2 * np.cos(a), h/3 + r2 * np.sin(a)))
        if len(config1) >= 11:
            break
    
    # Make sure all are inside triangle
    valid_config1 = []
    for x, y in config1:
        u, v = _cart_to_bary(x, y)
        if u >= -0.01 and v >= -0.01 and u + v <= 1.01:
            valid_config1.append([max(0, u), max(0, v)])
        else:
            # Clamp
            u = max(0, min(1, u))
            v = max(0, min(1 - u, v))
            valid_config1.append([u, v])
    
    if len(valid_config1) >= 11:
        initial_configs.append(valid_config1[:11])
    
    # Config 2: random with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(5):
        bary_rand = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            bary_rand.append([u, v])
        initial_configs.append(bary_rand)
    
    # Config 3: vertices and midpoints
    config3 = [
        [0.01, 0.01], [0.98, 0.01], [0.49, 0.49],  # near vertices
        [0.5, 0.01], [0.25, 0.43], [0.75, 0.43],  # midpoints of edges
        [1/3, 1/3],  # center
        [0.25, 0.25], [0.75, 0.25], [0.5, 0.5], [0.33, 0.66]
    ]
    # Clamp
    clamped3 = []
    for u, v in config3:
        u = max(0, min(1, u))
        v = max(0, min(1 - u, v))
        clamped3.append([u, v])
    if len(clamped3) == 11:
        initial_configs.append(clamped3)
    
    # Multi-start Nelder-Mead optimization
    for config in initial_configs:
        if time.time() > DEADLINE:
            break
        
        # Flatten barycentric coords
        x0 = np.array([coord for point in config for coord in point])
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _negative_min_area_bary,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        # Clamp result
        x_opt = _clamp_to_triangle(result.x)
        score = -_negative_min_area_bary(x_opt)
        
        if score > best_score:
            best_score = score
            best_bary = x_opt
    
    # Additional restarts with random perturbations if time allows
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        rng2 = np.random.RandomState(123)
        num_restarts = min(20, int(remaining_time / 2))
        
        for _ in range(num_restarts):
            if time.time() > DEADLINE:
                break
            
            # Perturb the best solution
            if best_bary is not None:
                x0 = best_bary + rng2.normal(0, 0.1, size=2 * N)
            else:
                u = rng2.random(N)
                v = rng2.random(N)
                mask = u + v > 1
                u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
                x0 = np.empty(2 * N)
                x0[0::2] = u
                x0[1::2] = v
            
            x0 = _clamp_to_triangle(x0)
            
            try:
                result = minimize(
                    _negative_min_area_bary,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                continue
            
            x_opt = _clamp_to_triangle(result.x)
            score = -_negative_min_area_bary(x_opt)
            
            if score > best_score:
                best_score = score
                best_bary = x_opt
    
    # Convert best barycentric to cartesian
    if best_bary is not None:
        h = (3 ** 0.5) / 2
        u = best_bary[0::2]
        v = best_bary[1::2]
        xs = u + 0.5 * v
        ys = v * h
        points = np.column_stack([xs, ys])
    else:
        # Fallback: random points
        rng3 = np.random.RandomState(0)
        points = []
        while len(points) < N:
            u, v = rng3.random(), rng3.random()
            if u + v < 1:
                points.append([u + 0.5 * v, v * (3 ** 0.5) / 2])
        points = np.array(points)
    
    return points
