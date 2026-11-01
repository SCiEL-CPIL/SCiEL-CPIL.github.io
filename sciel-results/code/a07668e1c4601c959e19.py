"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _compute_covering_radius(points):
    """
    Compute the exact covering radius using the spherical Delaunay triangulation
    (via 3D Convex Hull).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return float('inf')

    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        a11 = np.dot(v1, v1)
        a12 = np.dot(v1, v2)
        a22 = np.dot(v2, v2)
        b1 = 0.5 * a11
        b2 = 0.5 * a22
        
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            continue
            
        t1 = (b1 * a22 - b2 * a12) / det
        t2 = (a11 * b2 - a12 * b1) / det
        
        O = p1 + t1 * v1 + t2 * v2
        norm_O = np.linalg.norm(O)
        if norm_O < 1e-10:
            continue
        C = O / norm_O
        
        cos_theta = np.clip(np.dot(C, p1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def _repulsive_force(points, power=10.0, softening=1e-4):
    """
    Calculate the repulsive force on each point.
    """
    forces = np.zeros_like(points)
    n = len(points)
    
    for i in range(n):
        pi = points[i]
        for j in range(n):
            if i == j: continue
            d_vec = pi - points[j]
            d = np.linalg.norm(d_vec)
            if d < softening:
                d = softening
            # Force = k * (pi - pj) / d^(k+2)
            f_vec = power * d_vec / (d ** (power + 2))
            forces[i] += f_vec
            
    # Project forces onto tangent plane
    for i in range(n):
        n_vec = points[i]
        dot = np.dot(forces[i], n_vec)
        forces[i] -= dot * n_vec
        
    return forces

def _optimize_langevin(points, time_limit=50.0):
    """
    Optimize points using Langevin dynamics with cooling.
    """
    points = points.copy()
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    T = 0.1 
    alpha = 0.999 
    step_size = 0.01
    
    n = len(points)
    
    while time.time() - start_time < time_limit:
        forces = _repulsive_force(points, power=10.0)
        
        noise = np.random.randn(n, 3) * T
        points = points + step_size * (forces + noise)
        
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        T *= alpha
        
        if T < 1e-6:
            break
            
    return points

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_radius = float('inf')
    
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random Configurations
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    for idx, init_pts in enumerate(configs):
        if time.time() - start_time > time_limit:
            break
            
        remaining_time = time_limit - (time.time() - start_time)
        langevin_time = min(20.0, remaining_time * 0.8)
        
        optimized_pts = _optimize_langevin(init_pts, time_limit=langevin_time)
        
        radius = _compute_covering_radius(optimized_pts)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized_pts.copy()
            
    if best_points is None:
        best_points = configs[0]
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
