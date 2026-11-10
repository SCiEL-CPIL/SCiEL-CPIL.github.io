import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 580.0

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _outer_side_for(inner, outer_center, outer_angle):
    a = np.deg2rad(outer_angle)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(th), np.sin(th)])
    verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, 0.0) - outer_center for h in inner])
    max_proj = (verts @ normals.T).max(axis=0)
    return float((2.0 / np.sqrt(3.0)) * max_proj.max())

def _validity(inner, outer_center, outer_angle, side):
    if not np.all(np.isfinite(inner)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # Separation: center distance > sqrt(3) - epsilon
    # For unit hexagons, min center distance is sqrt(3).
    for i in range(11):
        for j in range(i + 1, 11):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < np.sqrt(3.0) - 1e-6:
                return False
    # Containment
    ov = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    verts = [_hex_vertices(h[0], h[1], 1.0, 0.0) for h in inner]
    for v in verts:
        for k in range(6):
            p1, p2 = ov[k], ov[(k + 1) % 6]
            if np.cross(p2 - p1, v - p1) < -1e-6:
                return False
    return True

def _initial_packing(combo):
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [d * np.cos(np.deg2rad(60.0 * k)), d * np.sin(np.deg2rad(60.0 * k)), 0.0]
    for k in range(4):
        ang = np.deg2rad(combo[k])
        inner[7 + k] = [2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang), 0.0]
    return inner

def _objective(params, outer_center):
    inner_xy = params[:22].reshape(11, 2)
    outer_angle = params[22]
    inner = np.column_stack((inner_xy, np.zeros(11)))
    
    side = _outer_side_for(inner, outer_center, outer_angle)
    
    # Fast penalty
    pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            dx = inner_xy[i, 0] - inner_xy[j, 0]
            dy = inner_xy[i, 1] - inner_xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < np.sqrt(3.0):
                pen += (np.sqrt(3.0) - dist) ** 2
    
    return side + 100.0 * pen

def hexagon_packing_11():
    oc0 = np.array([0.0, 0.0])
    best_inner = None
    best_side = np.inf
    best_ang = 0.0
    
    combinations = [
        [0, 60, 120, 180],
        [0, 180, 60, 120],
        [0, 120, 240, 60],
        [0, 60, 240, 120],
        [0, 60, 120, 240]
    ]
    
    for combo in combinations:
        if time.time() - _START_TIME > _TIME_LIMIT - 60.0:
            break
        try:
            inner0 = _initial_packing(combo)
            x0 = np.concatenate([inner0[:, :2].ravel(), [0.0]])
            
            res = minimize(_objective, x0, args=(oc0,), method="Nelder-Mead", 
                           options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
            
            inner_opt = res.x[:22].reshape(11, 2)
            ang_opt = res.x[22]
            inner_full = np.column_stack((inner_opt, np.zeros(11)))
            
            side_opt = _outer_side_for(inner_full, oc0, ang_opt)
            
            if _validity(inner_full, oc0, ang_opt, side_opt):
                if side_opt < best_side:
                    best_inner = inner_full.copy()
                    best_side = side_opt
                    best_ang = ang_opt
        except Exception:
            continue
            
    if best_inner is None:
        inner0 = _initial_packing([0, 60, 120, 180])
        s0 = _outer_side_for(inner0, oc0, 0.0)
        best_inner = inner0.copy()
        best_side = s0
        best_ang = 0.0

    return best_inner, np.array([oc0[0], oc0[1], best_ang]), float(best_side)
