import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _spherical_circumradius(p1, p2, p3):
    v1 = p1 - p2
    v2 = p1 - p3
    c_dir = np.cross(v1, v2)
    norm_c = np.linalg.norm(c_dir)
    if norm_c < 1e-15:
        return 0.0
    C = c_dir / norm_c
    n = np.cross(p2 - p1, p3 - p1)
    if np.dot(C, n) < 0:
        C = -C
    return np.arccos(np.clip(np.dot(C, p1), -1, 1))


def _spherical_circumcenter(p1, p2, p3):
    v1 = p1 - p2
    v2 = p1 - p3
    c_dir = np.cross(v1, v2)
    norm_c = np.linalg.norm(c_dir)
    if norm_c < 1e-15:
        return None
    C = c_dir / norm_c
    n = np.cross(p2 - p1, p3 - p1)
    if np.dot(C, n) < 0:
        C = -C
    return C


def _covering_radius(pts):
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for tri in hull.simplices:
            r = _spherical_circumradius(pts[tri[0]], pts[tri[1]], pts[tri[2]])
            if r > max_r:
                max_r = r
        return np.degrees(max_r)
    except:
        return 180.0


def _optimize(pts, deadline, max_iters=5000):
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _covering_radius(best_pts)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(pts)
            simplices = hull.simplices
        except:
            break
        
        # Find worst facet
        radii = np.array([_spherical_circumradius(pts[t[0]], pts[t[1]], pts[t[2]]) for t in simplices])
        max_idx = np.argmax(radii)
        tri = simplices[max_idx]
        
        C = _spherical_circumcenter(pts[tri[0]], pts[tri[1]], pts[tri[2]])
        if C is None:
            break
        
        # Annealing schedule
        progress = it / max_iters
        alpha = 0.03 * (1.0 - progress) ** 0.5
        
        # Move vertices toward circumcenter
        for vi in tri:
            direction = C - pts[vi]
            norm_d = np.linalg.norm(direction)
            if norm_d > 1e-15:
                pts[vi] += alpha * direction / norm_d
        
        # Renormalize
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        r = _covering_radius(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    best_pts, best_r = _optimize(pts, deadline)
    
    return best_pts

