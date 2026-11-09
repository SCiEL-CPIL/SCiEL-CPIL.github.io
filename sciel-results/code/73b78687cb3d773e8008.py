import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    start_time = time.time()
    deadline = start_time + 590

    def objective(x):
        if time.time() > deadline:
            return np.inf
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        try:
            dists = pdist(pts)
        except Exception:
            return np.inf
            
        if np.any(~np.isfinite(dists)):
            return np.inf
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return np.inf
            
        ratio = (d_min / d_max) ** 2
        return -ratio

    def icosahedron():
        """Return 12 vertices of regular icosahedron on unit sphere."""
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    ico_base = icosahedron()

    def build_ico_poles(params):
        """
        params: [r_ico, z_pole]
        """
        r_ico = params[0]
        z_pole = params[1]
        
        ico_pts = ico_base * r_ico
        poles = np.array([[0, 0, z_pole], [0, 0, -z_pole]])
        
        pts = np.vstack([ico_pts, poles])
        return pts

    def build_rings_poles(params):
        """
        params: [r1, z1, r2, z2, z3]
        Ring 1: 6 points, radius r1, height z1, angles 0, 60, ...
        Ring 2: 6 points, radius r2, height z2, angles 30, 90, ...
        Poles: 2 points, height +/- z3
        """
        r1, z1, r2, z2, z3 = params
        
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        ring1 = np.column_stack([
            r1 * np.cos(angles1),
            r1 * np.sin(angles1),
            np.full(6, z1)
        ])
        
        angles2 = angles1 + np.pi/6
        ring2 = np.column_stack([
            r2 * np.cos(angles2),
            r2 * np.sin(angles2),
            np.full(6, z2)
        ])
        
        poles = np.array([[0, 0, z3], [0, 0, -z3]])
        
        pts = np.vstack([ring1, ring2, poles])
        return pts

    def objective_ico_poles(x):
        if time.time() > deadline:
            return np.inf
        pts = build_ico_poles(x)
        return objective(pts)

    def objective_rings_poles(x):
        if time.time() > deadline:
            return np.inf
        pts = build_rings_poles(x)
        return objective(pts)

    best_val = np.inf
    best_pts = None

    # Optimization 1: Icosahedron + Poles
    ico_guesses = [
        [1.0, 1.2],
        [1.0, 0.8],
        [1.2, 1.0],
        [0.8, 1.5]
    ]
    bounds_ico = [(0.1, 3.0), (-3.0, 3.0)]
    
    for x0 in ico_guesses:
        if time.time() > deadline - 20:
            break
        try:
            res_ico = minimize(objective_ico_poles, x0, bounds=bounds_ico, method='L-BFGS-B', options={'maxiter': 500})
            if res_ico.fun < best_val and np.isfinite(res_ico.fun):
                best_val = res_ico.fun
                best_pts = build_ico_poles(res_ico.x)
        except Exception:
            pass

    # Optimization 2: Rings + Poles
    rings_guesses = [
        [1.0, 0.5, 1.0, -0.5, 1.5],
        [1.0, 0.3, 1.0, -0.3, 1.2],
        [1.2, 0.6, 0.8, -0.6, 1.8]
    ]
    bounds_rings = [(0.1, 3.0), (-3.0, 3.0), (0.1, 3.0), (-3.0, 3.0), (0.1, 3.0)]
    
    for x0 in rings_guesses:
        if time.time() > deadline - 20:
            break
        try:
            res_rings = minimize(objective_rings_poles, x0, bounds=bounds_rings, method='L-BFGS-B', options={'maxiter': 500})
            if res_rings.fun < best_val and np.isfinite(res_rings.fun):
                best_val = res_rings.fun
                best_pts = build_rings_poles(res_rings.x)
        except Exception:
            pass

    if best_pts is None:
        best_pts = build_ico_poles([1.0, 1.2])

    final_points = best_pts - np.mean(best_pts, axis=0)
    return final_points
