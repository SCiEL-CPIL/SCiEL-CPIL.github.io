import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array([(cx + side * np.cos(angle_rad + k * np.pi / 3),
                      cy + side * np.sin(angle_rad + k * np.pi / 3)) for k in range(6)])

def all_hex_vertices(inner, side=1.0):
    return np.vstack([hex_vertices(x, y, side, ang) for (x, y, ang) in inner])

def min_enclosing_hex_side(verts, center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = np.array([(np.cos(angle_rad + k * np.pi / 3),
                         np.sin(angle_rad + k * np.pi / 3)) for k in range(6)])
    rel = verts - np.array(center)
    proj = rel @ normals.T
    return np.maximum(proj.max(axis=0), -proj.min(axis=0)).max() / (np.sqrt(3) / 2)

def objective(params, verts):
    return min_enclosing_hex_side(verts, (params[0], params[1]), params[2])

def optimize_outer(inner, time_budget=3.0):
    verts = all_hex_vertices(inner)
    centroid = verts.mean(axis=0)
    t0 = time.time()
    best_side, best_params = float('inf'), [centroid[0], centroid[1], 0.0]
    for init_angle in [0, 30, 60, 90, 120, 150]:
        if time.time() - t0 > time_budget:
            break
        try:
            res = minimize(objective, [centroid[0], centroid[1], float(init_angle)], args=(verts,),
                          method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_side:
                best_side, best_params = res.fun, res.x
        except Exception:
            pass
    return best_params, best_side

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    sqrt3 = np.sqrt(3.0)
    a1, a2 = np.array([sqrt3, 0.0]), np.array([sqrt3 / 2.0, 1.5])
    
    def hex_dist(q1, r1, q2, r2):
        return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))
    
    all_pts = sorted(
        [((q, r), q * a1 + r * a2) for q in range(-4, 5) for r in range(-4, 5) if hex_dist(0, 0, q, r) <= 3],
        key=lambda x: hex_dist(0, 0, x[0][0], x[0][1]))
    
    candidates = [list(set([p[0] for p in all_pts[:11]]))]
    for i in range(11, min(25, len(all_pts))):
        c = [p[0] for p in all_pts[:11]]
        c[i - 11] = all_pts[i][0]
        candidates.append(list(set(c)))
    candidates.append(list(set([(q, r) for r in [-1, 0, 1] for q in range(-2, 2)]))[:11])
    candidates.append(list(set([(q, 0) for q in range(-3, 3)] + [(q, 1) for q in range(-3, 2)])))
    candidates.append(list(set([(q, 0) for q in range(-5, 6)])))
    
    best_side, best_inner, best_outer = float('inf'), None, None
    for cset in candidates:
        if time.time() > deadline - 15 or len(cset) < N:
            break
        inner = [[float((q * a1 + r * a2)[0]), float((q * a1 + r * a2)[1]), 0.0] for (q, r) in cset[:N]]
        try:
            params, side = optimize_outer(inner, time_budget=2.0)
            if side < best_side:
                best_side, best_inner, best_outer = side, inner, params
        except Exception:
            pass
    
    if best_inner is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    cx, cy, angle = best_outer
    return best_inner, (float(cx), float(cy)), float(best_side), float(angle)
