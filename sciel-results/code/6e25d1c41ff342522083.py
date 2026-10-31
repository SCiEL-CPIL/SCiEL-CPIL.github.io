"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    return 0.5 * abs(np.cross(q - p, r - p))

def _min_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_a = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _neg_min_area(points):
    """Negative of the minimum area for minimization."""
    return -_min_area(points)

def _in_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    # Barycentric coordinates
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    den = d00 * d11 - d01 * d01
    if den == 0:
        return False
    v = (d11 * d02 - d01 * d12) / den
    w = (d00 * d12 - d01 * d02) / den
    u = 1.0 - v - w
    return u >= -1e-9 and v >= -1e-9 and w >= -1e-9

def _clamp_to_triangle(p):
    """Clamp point to be inside the triangle (approximate)."""
    # Simple approach: if outside, move towards centroid
    if _in_triangle(p):
        return p
    # Move towards centroid
    centroid = (V0 + V1 + V2) / 3
    t = 0.5
    while t < 1.0:
        candidate = p * (1 - t) + centroid * t
        if _in_triangle(candidate):
            return candidate
        t += 0.1
    return centroid

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial best solution
    best_points = None
    best_value = -np.inf
    
    # Strategy 1: Start with a symmetric configuration
    # Place points in a grid-like pattern inside the triangle
    def _initial_grid():
        points = []
        # Center
        points.append((V0 + V1 + V2) / 3)
        # Midpoints of edges
        points.append((V0 + V1) / 2)
        points.append((V1 + V2) / 2)
        points.append((V2 + V0) / 2)
        # Centroid and some interior points
        # Add 7 more points in a hexagonal pattern around center
        center = (V0 + V1 + V2) / 3
        r = 0.15
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            p = center + r * np.array([np.cos(angle), np.sin(angle)])
            if _in_triangle(p):
                points.append(p)
        # If we have more than 11, trim; if less, add random
        points = points[:11]
        while len(points) < 11:
            # Add random points
            u, v = np.random.rand(2)
            if u + v < 1:
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                points.append(p)
        return np.array(points)
    
    # Strategy 2: Random initialization
    def _random_init():
        points = []
        while len(points) < N:
            u, v = np.random.rand(2)
            if u + v < 1:
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                points.append(p)
        return np.array(points)
    
    # Try several initial configurations
    inits = [_initial_grid(), _random_init(), _random_init()]
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Local optimization using Nelder-Mead
        # Flatten points to 1D
        x0 = init_points.flatten()
        
        def _objective(x):
            points = x.reshape(-1, 2)
            # Check if all points are inside triangle
            for p in points:
                if not _in_triangle(p):
                    return 1e10  # Large penalty
            return -_min_area(points)
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if time.time() > deadline:
                break
            
            if result.success or _min_area(result.x.reshape(-1, 2)) > best_value:
                opt_points = result.x.reshape(-1, 2)
                # Clamp to triangle
                opt_points = np.array([_clamp_to_triangle(p) for p in opt_points])
                val = _min_area(opt_points)
                if val > best_value:
                    best_value = val
                    best_points = opt_points
        except Exception:
            pass
    
    # If we still don't have a good solution, use the initial grid
    if best_points is None:
        best_points = inits[0]
        best_value = _min_area(best_points)
    
    # Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10 and restarts < 5:
        restarts += 1
        random_points = _random_init()
        
        def _objective(x):
            points = x.reshape(-1, 2)
            for p in points:
                if not _in_triangle(p):
                    return 1e10
            return -_min_area(points)
        
        try:
            result = minimize(_objective, random_points.flatten(), method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if time.time() > deadline:
                break
            if result.success:
                opt_points = result.x.reshape(-1, 2)
                opt_points = np.array([_clamp_to_triangle(p) for p in opt_points])
                val = _min_area(opt_points)
                if val > best_value:
                    best_value = val
                    best_points = opt_points
        except Exception:
            pass
    
    # Final check: ensure all points are inside triangle
    final_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return final_points
# EVOLVE-BLOCK-END
