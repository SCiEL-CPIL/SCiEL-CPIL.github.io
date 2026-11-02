import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, angle_rad, side=1.0):
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hexagon(p, cx, cy, angle_rad, side=1.0):
    verts = _hex_vertices(cx, cy, angle_rad, side)
    inside = True
    min_dist = np.inf
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        v = p2 - p1
        w = p - p1
        cross = v[0] * w[1] - v[1] * w[0]
        if cross < 0:
            inside = False
        # Calculate distance to segment
        t = np.dot(w, v) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        proj = p1 + t * v
        dist = np.linalg.norm(p - proj)
        if dist < min_dist:
            min_dist = dist
    if inside:
        return 0.0
    else:
        return min_dist

def _outer_hex_penalty(p, S):
    R_in = S * np.sqrt(3) / 2
    penalty = 0.0
    for k in range(6):
        angle = np.pi / 6 + k * np.pi / 3
        nx = np.cos(angle)
        ny = np.sin(angle)
        val = nx * p[0] + ny * p[1] - R_in
        if val > 0:
            penalty += val * val
    return penalty

def _objective(params):
    S = params[-1]
    if S < 2.0:
        return 1e6
    
    inner_params = params[:-1]
    total_penalty = 0.0
    
    # 1. Containment
    for i in range(N):
        x, y, th = inner_params[i*3:(i+1)*3]
        verts = _hex_vertices(x, y, th)
        for v in verts:
            total_penalty += _outer_hex_penalty(v, S)
            
    # 2. Overlap
    for i in range(N):
        x1, y1, th1 = inner_params[i*3:(i+1)*3]
        for j in range(i+1, N):
            x2, y2, th2 = inner_params[j*3:(j+1)*3]
            dist_centers = np.linalg.norm([x1-x2, y1-y2])
            if dist_centers > 2.0:
                continue
            verts_i = _hex_vertices(x1, y1, th1)
            for v in verts_i:
                d = _point_in_hexagon(v, x2, y2, th2)
                if d > 0:
                    total_penalty += d * d
            verts_j = _hex_vertices(x2, y2, th2)
            for v in verts_j:
                d = _point_in_hexagon(v, x1, y1, th1)
                if d > 0:
                    total_penalty += d * d
                    
    return S + 100.0 * total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess
    x0 = []
    y0 = []
    th0 = []
    
    # Center
    x0.append(0.0); y0.append(0.0); th0.append(0.0)
    
    # 6 around
    for k in range(6):
        angle = k * np.pi / 3
        x0.append(np.sqrt(3) * np.cos(angle))
        y0.append(np.sqrt(3) * np.sin(angle))
        th0.append(angle)
        
    # 4 more
    # Place them at the "corners" of the 7-hexagon cluster
    # Distance from center: sqrt(3) + sqrt(3) = 2*sqrt(3)?
    # Let's just place them at specific spots
    x0.append(2*np.sqrt(3)); y0.append(0.0); th0.append(0.0)
    x0.append(-2*np.sqrt(3)); y0.append(0.0); th0.append(0.0)
    x0.append(np.sqrt(3)); y0.append(3.0); th0.append(0.0)
    x0.append(-np.sqrt(3)); y0.append(3.0); th0.append(0.0)
    
    S0 = 4.0
    params0 = np.array(x0 + y0 + th0 + [S0])
    
    best_params = params0
    best_score = _objective(params0)
    
    # Optimization
    try:
        res = minimize(_objective, params0, method='Powell', 
                       options={'maxiter': 1000, 'xtol': 1e-4, 'ftol': 1e-4})
        if time.time() < deadline:
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
    except Exception:
        pass

    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(_objective, best_params, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass

    S = best_params[-1]
    inner = []
    for i in range(N):
        x, y, th = best_params[i*3:(i+1)*3]
        inner.append([float(x), float(y), float(np.rad2deg(th))])
        
    return inner, (0.0, 0.0), float(S), 0.0
