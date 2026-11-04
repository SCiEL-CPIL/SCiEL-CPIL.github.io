import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices(x, y, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([x + np.cos(angles), y + np.sin(angles)])

def _outer_side_length(pos, angles_deg, cx, cy, outer_angle_deg):
    outer_a = np.deg2rad(outer_angle_deg)
    normals = np.column_stack([np.cos(outer_a + np.arange(6) * np.pi / 3),
                                np.sin(outer_a + np.arange(6) * np.pi / 3)])
    max_proj = 0.0
    for i in range(len(pos)):
        verts = _hex_vertices(pos[i][0], pos[i][1], angles_deg[i])
        rel = verts - np.array([cx, cy])
        projs = rel @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    return max_proj / (np.sqrt(3) / 2)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    sp = np.sqrt(3)
    axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
             (2,0), (1,1), (-1,2), (-2,1)]
    
    def axial_to_cart(q, r):
        x = sp * (q + r/2.0)
        y = sp * (np.sqrt(3)/2.0) * r
        return x, y
    
    init_positions = [axial_to_cart(q, r) for q, r in axial]
    
    def full_objective(params):
        if time.time() > deadline:
            return 1e10
        pos = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_rot = params[3*N] * 180.0 / np.pi
        return _outer_side_length(pos, angles, 0.0, 0.0, outer_rot)
    
    x0 = np.array([v for p in init_positions for v in p] + [0.0]*N + [0.0])
    
    best_score = full_objective(x0)
    best_params = x0.copy()
    
    try:
        result = minimize(full_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time.time() > deadline:
        pos = best_params[:2*N].reshape(N, 2)
        angles = best_params[2*N:3*N] * 180.0 / np.pi
        return [[pos[i][0], pos[i][1], angles[i]] for i in range(N)], (0.0, 0.0), best_score, 0.0
    
    # Multi-start with random perturbations
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline:
            break
        x0_trial = x0 + rng.normal(0, 0.5, size=x0.shape)
        try:
            res = minimize(full_objective, x0_trial, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    pos = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:3*N] * 180.0 / np.pi
    return [[pos[i][0], pos[i][1], angles[i]] for i in range(N)], (0.0, 0.0), best_score, 0.0
