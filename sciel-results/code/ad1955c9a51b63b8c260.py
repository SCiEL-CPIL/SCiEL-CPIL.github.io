import numpy as np
import time
from scipy.optimize import minimize
import math

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a flower packing (1 center, 6 ring 1, 4 ring 2) with optimized rotations.
    """
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in the standard hex lattice
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    def _get_rotated_vertices(phi):
        """Get rotated vertices for a unit hexagon with rotation phi."""
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        return base_vertices @ rot.T
    
    def compute_outer_side(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Vectorized computation.
        """
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        # Pre-compute rotated vertices for all hexagons
        all_verts = []
        for i in range(n_hex):
            verts = _get_rotated_vertices(phis[i]) + centers[i]
            all_verts.append(verts)
        all_verts = np.array(all_verts)  # shape (n_hex, 6, 2)
        
        # Compute projections: (n_hex, 6, 2) @ (2, 6) -> (n_hex, 6, 6)
        projs = np.einsum('ijk,jl->ikl', all_verts, normals)
        max_proj = np.max(projs)
        
        return 2.0 * max_proj / np.sqrt(3)

    def objective_smooth(params, beta=50.0):
        """
        Smooth objective using log-sum-exp approximation.
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        all_verts = []
        for i in range(n_hex):
            verts = _get_rotated_vertices(phis[i]) + centers[i]
            all_verts.append(verts)
        all_verts = np.array(all_verts)
        
        projs = np.einsum('ijk,jl->ikl', all_verts, normals)
        flat_projs = projs.flatten()
        
        m = np.max(flat_projs)
        soft_max = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (flat_projs - m))))
        
        return 2.0 * soft_max / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)
    
    # Initial guess: align each hexagon's face normal radially outward
    x0 = np.zeros(n_hex + 1)
    x0[0] = 0.0
    for i in range(n_hex):
        cx, cy = centers[i]
        if np.abs(cx) < 1e-9 and np.abs(cy) < 1e-9:
            x0[i+1] = 0.0
        else:
            alpha = np.arctan2(cy, cx)
            x0[i+1] = alpha - np.deg2rad(30)
            
    best_score = float('inf')
    best_params = None
    
    # Multi-start optimization
    n_starts = 15
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            x_start = np.zeros(n_hex + 1)
        elif i == 2:
            x_start = x0.copy()
            x_start[0] = np.deg2rad(30)
        elif i == 3:
            x_start = x0.copy()
            x_start[0] = np.deg2rad(15)
        elif i == 4:
            x_start = x0.copy()
            x_start[0] = np.deg2rad(45)
        else:
            x_start = x0 + np.random.uniform(-0.2, 0.2, n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
            
        # Smooth phase
        try:
            res_smooth = minimize(objective_smooth, x_start, method='L-BFGS-B',
                                  options={'maxiter': 200, 'disp': False, 'ftol': 1e-10})
            x_start = res_smooth.x
        except Exception:
            pass
        
        # Exact refinement with Powell
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 400, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return _fallback_solution()
    
    # Nelder-Mead polish
    if time.time() < deadline - 8:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 3000, 'disp': False})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
            
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(np.rad2deg(phis[i]))])
    
    # Add small buffer to ensure validity
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), float(side), float(np.rad2deg(theta_out))

def _fallback_solution():
    """
    Valid fallback: standard hex lattice with all hexagons aligned.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 30.0])
    
    # For the standard lattice with all hexagons at 30 degrees,
    # the outer hexagon needs side length 4 (apothem = 2*sqrt(3))
    side = 4.0 + 1e-8
    return inner, (0.0, 0.0), float(side), 30.0
