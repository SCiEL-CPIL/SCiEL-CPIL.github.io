import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Hexagonal lattice for unit hexagons (side 1), flat-top
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    # 11 lattice points in compact cluster
    lattice_indices = [
        (0, 0),       # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # ring 1
        (2, 0), (-2, 0), (0, 2), (0, -2)  # 4 from ring 2
    ]
    
    centers = []
    for i, j in lattice_indices:
        c = i * a1 + j * a2
        centers.append(c)
    centers = np.array(centers)
    
    # All vertices of all inner hexagons (unit, flat-top, rotation 0)
    all_verts = []
    for c in centers:
        for k in range(6):
            a = k * np.pi / 3.0
            v = c + np.array([np.cos(a), np.sin(a)])
            all_verts.append(v)
    all_verts = np.array(all_verts)
    
    def objective(params):
        cx, cy, theta_deg = params
        theta = np.radians(theta_deg)
        normals = np.array([[np.cos(theta + k * np.pi / 3.0),
                             np.sin(theta + k * np.pi / 3.0)] for k in range(6)])
        rel = all_verts - np.array([cx, cy])
        dots = rel @ normals.T
        max_dots = dots.max(axis=0)
        s = (2.0 / np.sqrt(3.0)) * max_dots.max()
        return s
    
    best_s = np.inf
    best_params = None
    
    # Multi-start over rotation
    for theta0 in range(0, 360, 10):
        if time.time() > deadline:
            break
        cx0 = all_verts[:, 0].mean()
        cy0 = all_verts[:, 1].mean()
        try:
            res = minimize(objective, [cx0, cy0, float(theta0)],
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final refinement
    if best_params is not None and time.time() < deadline:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    cx, cy, theta_deg = best_params
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    
    return inner, (float(cx), float(cy)), float(best_s), float(theta_deg)
