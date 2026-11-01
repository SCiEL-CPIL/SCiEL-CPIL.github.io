import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_deg(config):
    """
    Compute the covering radius in degrees using convex hull facets.
    The covering radius is the maximum over all hull facets of the angular
    distance from the facet's circumcenter (if valid) or the worst vertex
    to the nearest configuration point.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback: dense sampling
        grid = _fib_sample(30000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    max_angle = 0.0
    for simplex in hull.simplices:
        pts = config[simplex]  # (3, 3)
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        
        if c_norm < 1e-12:
            # Degenerate facet
            for p in pts:
                dots = np.clip(config @ p, -1.0, 1.0)
                ang = np.arccos(np.max(dots))
                if ang > max_angle:
                    max_angle = ang
            continue
        
        c = c / c_norm
        
        # Check if circumcenter is valid (equidistant from all 3 vertices)
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        if np.max(angles) - np.min(angles) > 1e-5:
            # Circumcenter is outside the triangle; worst point is a vertex
            for p in pts:
                dots_all = np.clip(config @ p, -1.0, 1.0)
                ang = np.arccos(np.max(dots_all))
                if ang > max_angle:
                    max_angle = ang
        else:
            # Circumcenter is valid and inside the triangle
            dots_all = np.clip(config @ c, -1.0, 1.0)
            ang = np.arccos(np.max(dots_all))
            if ang > max_angle:
                max_angle = ang
    
    return np.degrees(max_angle)

def _lloyd_step(config, grid):
    """One step of Lloyd's algorithm using a fixed grid."""
    dots = grid @ config.T
    max_idx = np.argmax(dots, axis=1)
    
    new_config = np.zeros_like(config)
    for i in range(N):
        mask = (max_idx == i)
        count = np.sum(mask)
        if count > 0:
            center = np.mean(grid[mask], axis=0)
            new_config[i] = _normalize(center)
        else:
            new_config[i] = _normalize(np.random.randn(3))
    return new_config

def _refine(config, deadline):
    """
    Refine the configuration using a local search on the covering radius.
    Uses hull-based critical points to guide moves.
    """
    start_time = time.time()
    config = config.copy()
    
    current_score = _covering_radius_deg(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.03
    stagnation = 0
    
    while time.time() < deadline - 15:
        # Find the worst point on the sphere using hull facets
        try:
            hull = ConvexHull(config)
        except Exception:
            break
        
        worst_angle = 0.0
        worst_point = None
        
        for simplex in hull.simplices:
            pts = config[simplex]
            
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            c = np.cross(v1, v2)
            c_norm = np.linalg.norm(c)
            
            if c_norm < 1e-12:
                continue
            
            c = c / c_norm
            
            dots = np.clip(pts @ c, -1.0, 1.0)
            angles = np.arccos(dots)
            
            if np.max(angles) - np.min(angles) > 1e-5:
                for idx in simplex:
                    p = config[idx]
                    dots_all = np.clip(config @ p, -1.0, 1.0)
                    ang = np.arccos(np.max(dots_all))
                    if ang > worst_angle:
                        worst_angle = ang
                        worst_point = p
            else:
                dots_all = np.clip(config @ c, -1.0, 1.0)
                ang = np.arccos(np.max(dots_all))
                if ang > worst_angle:
                    worst_angle = ang
                    worst_point = c
        
        if worst_point is None:
            break
        
        # Try to improve by moving the nearest point toward the worst point
        dots = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots)
        
        moved = False
        for idx in sorted_idx[:5]:  # Try top 5 nearest
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            new_pt = _normalize(config[idx] + step_size * v_proj / v_norm)
            old_pt = config[idx].copy()
            config[idx] = new_pt
            
            new_score = _covering_radius_deg(config)
            
            if new_score < current_score - 1e-10:
                current_score = new_score
                stagnation = 0
                if new_score < best_score:
                    best_score = new_score
                    best_config = config.copy()
                moved = True
            else:
                config[idx] = old_pt
        
        if not moved:
            stagnation += 1
            step_size *= 0.92
            if step_size < 0.0005:
                # Try random perturbation
                idx = np.random.randint(N)
                old_pt = config[idx].copy()
                config[idx] = _normalize(config[idx] + 0.02 * np.random.randn(3))
                new_score = _covering_radius_deg(config)
                if new_score < current_score:
                    current_score = new_score
                    stagnation = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                else:
                    config[idx] = old_pt
                if stagnation > 60:
                    break
    
    return best_config

def _optimize_from_start(config, deadline):
    """Full optimization pipeline from a starting configuration."""
    start_time = time.time()
    
    # Lloyd's algorithm with a grid
    grid = _fib_sample(10000)
    
    for _ in range(50):
        if time.time() > deadline - 80:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-9):
            break
        config = new_config
    
    # Refine
    config = _refine(config, deadline)
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    np.random.seed(42)
    
    best_config = None
    best_score = float('inf')
    
    # Try multiple random restarts
    num_restarts = 3
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        if restart == 0:
            config = _fib_sample(N)
        else:
            # Random start with some structure
            config = np.random.randn(N, 3)
            config = np.array([_normalize(p) for p in config])
            # Ensure spread: reject if all in a hemisphere
            for attempt in range(10):
                if np.min(np.diag(np.linalg.multi_dot([config.T, config]))) < -0.5:
                    break
                config = np.array([_normalize(p + 0.1 * np.random.randn(3)) for p in config])
        
        config = _optimize_from_start(config, restart_deadline)
        
        score = _covering_radius_deg(config)
        if score < best_score:
            best_score = score
            best_config = config.copy()
    
    # Final polish on the best configuration if time remains
    if best_config is not None and time.time() < deadline - 20:
        best_config = _refine(best_config, deadline)
    
    # Final normalization and validation
    if best_config is not None:
        best_config = np.array([_normalize(p) for p in best_config])
        
        # Ensure no coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_config[i] - best_config[j]) < 1e-6:
                    best_config[j] = best_config[j] + 1e-4 * np.random.randn(3)
                    best_config[j] = _normalize(best_config[j])
    
    return best_config
