# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_circumradius(u, v, w):
    """
    Compute the circumradius of the spherical triangle formed by u, v, w.
    Returns the angular radius in radians.
    """
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)
    
    d_uv = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    d_vw = np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))
    d_wu = np.arccos(np.clip(np.dot(w, u), -1.0, 1.0))
    
    sin_a = np.sin(d_uv)
    sin_b = np.sin(d_vw)
    sin_c = np.sin(d_wu)
    
    cos_a = np.cos(d_uv)
    cos_b = np.cos(d_vw)
    cos_c = np.cos(d_wu)
    
    denom = cos_a + cos_b + cos_c - 1.0
    
    if np.abs(denom) < 1e-10:
        return np.max([d_uv, d_vw, d_wu]) / 2.0
        
    tan_R = (sin_a * sin_b * sin_c) / denom
    R_c = np.arctan(tan_R)
    
    return R_c

def _covering_radius(points):
    """
    Compute the covering radius of the set of points on the sphere.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2.0
        
    max_rad = 0.0
    for simplex in hull.simplices:
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]
        
        R_c = _spherical_circumradius(u, v, w)
        
        d_uv = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
        d_vw = np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))
        d_wu = np.arccos(np.clip(np.dot(w, u), -1.0, 1.0))
        max_half_edge = np.max([d_uv, d_vw, d_wu]) / 2.0
        
        facet_rad = np.max([R_c, max_half_edge])
        
        if facet_rad > max_rad:
            max_rad = facet_rad
            
    return max_rad

def _objective(flat):
    """
    Objective function for optimization.
    flat: (3*N,) array.
    """
    points = flat.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    rad = _covering_radius(points)
    return rad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_score = _covering_radius(points_init)
    best_points = points_init.copy()
    
    from scipy.optimize import minimize
    
    # Multi-start optimization
    for restart in range(5):
        if time.time() > deadline:
            break
            
        noise_scale = 0.1 if restart > 0 else 0.0
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, (N, 3))
            points_start = points_init + noise
            norms = np.linalg.norm(points_start, axis=1, keepdims=True)
            points_start = points_start / norms
        else:
            points_start = points_init.copy()
            
        flat_start = points_start.flatten()
        
        try:
            res = minimize(_objective, flat_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, 3)
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                best_points = best_points / norms
                
        except Exception:
            pass
            
    # Final polish
    if time.time() < deadline - 20:
        flat_start = best_points.flatten()
        try:
            res = minimize(_objective, flat_start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, 3)
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                best_points = best_points / norms
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
