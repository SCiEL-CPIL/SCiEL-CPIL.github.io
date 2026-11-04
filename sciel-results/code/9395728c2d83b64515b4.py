import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_verts(ang):
    th = ang + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(th), np.sin(th)])

def _hex_normals(ang):
    th = ang + (np.arange(6) + 0.5) * np.pi / 3
    return np.column_stack([np.cos(th), np.sin(th)])

def _gap(d1, d2, v1, v2, n1, n2):
    all_n = np.vstack([n1, n2])
    min_gap = np.inf
    for k in range(12):
        n = all_n[k]
        s1 = np.max(v1 @ n)
        s2 = np.max(v2 @ n)
        c1 = d1 @ n
        c2 = d2 @ n
        gap = min(c1 - s1, c2 - s2) - max(c1 + s1, c2 + s2)
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _outer_side(pos, ang):
    normals = _hex_normals(0.0)
    apothem = np.sqrt(3) / 2
    max_ext = 0.0
    for i in range(N):
        v = _hex_verts(ang[i])
        d = pos[i]
        for n in normals:
            s = np.max(v @ n)
            ext = d @ n + s
            if ext > max_ext:
                max_ext = ext
    return max_ext / apothem

def _objective(params):
    pos = params[:2*N].reshape(N, 2)
    ang = params[2*N:]
    
    verts = [_hex_verts(ang[i]) for i in range(N)]
    norms = [_hex_normals(ang[i]) for i in range(N)]
    
    side = _outer_side(pos, ang)
    
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            g = _gap(pos[i], pos[j], verts[i], verts[j], norms[i], norms[j])
            if g < 0:
                pen += g * g
    
    return side + 100.0 * pen

def construct():
    t0 = time.time()
    deadline = t0 + 125.0
    
    sqrt3 = np.sqrt(3)
    
    positions = []
    for r in range(3):
        for q in range(4):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            positions.append([x, y])
    
    pos_arr = np.array(positions)
    pos_arr -= pos_arr.mean(axis=0)
    
    angles = np.zeros(N)
    params = np.concatenate([pos_arr.flatten(), angles])
    
    best_params = params.copy()
    best_val = _objective(best_params)
    
    while time.time() < deadline:
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
        
        p = best_params + np.random.normal(0, 0.4, size=best_params.shape)
        p[2*N:] = np.mod(p[2*N:], 2*np.pi)
        try:
            res = minimize(_objective, p, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    pos = best_params[:2*N].reshape(N, 2)
    ang = best_params[2*N:]
    
    inner = [[float(pos[i][0]), float(pos[i][1]), float(np.degrees(ang[i]))] for i in range(N)]
    outer_s = _outer_side(pos, ang)
    
    return inner, (0.0, 0.0), float(outer_s), 0.0
