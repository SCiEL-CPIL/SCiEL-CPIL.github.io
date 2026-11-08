import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons_fast(v1, v2):
        min_d = float('inf')
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        return min_d

    def evaluate_violation(params, outer_s):
        xs = params[0:12]
        ys = params[12:24]
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, 0.0)
            inner_verts.append(v)
        
        violation = 0.0
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                out = np.maximum(0, dists - r)
                violation += np.sum(out)
        
        for i in range(12):
            for j in range(i+1, 12):
                cx1, cy1 = xs[i], ys[i]
                cx2, cy2 = xs[j], ys[j]
                c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                threshold = np.sqrt(3)
                if c_dist < threshold:
                    violation += (threshold - c_dist) ** 2 * 10.0
        return violation

    def objective(params):
        return evaluate_violation(params, current_outer_s)

    def get_initial_positions():
        pos = []
        for i in range(12):
            x = (i % 4) * 2.0 - 3.0
            y = (i // 4) * 1.8 - 1.5
            pos.append([x, y])
        return np.array(pos).flatten()

    LB = 2.5
    UB = 10.0
    best_solution = None
    best_s = float('inf')
    
    max_iter_nm = 200
    tol = 1e-4
    
    for step in range(30):
        if time.time() > deadline:
            break
        mid_s = (LB + UB) / 2.0
        current_outer_s = mid_s
        found_valid = False
        best_violation = float('inf')
        
        for restart in range(3):
            if time.time() > deadline:
                break
            x0 = get_initial_positions()
            x0 += np.random.normal(0, 0.5, x0.shape)
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_violation:
                    best_violation = res.fun
                    best_params = res.x
                if res.fun < 1e-3:
                    found_valid = True
                    best_params = res.x
                    break
            except Exception:
                pass
        
        if found_valid:
            UB = mid_s
            best_solution = best_params.copy()
            best_s = mid_s
        else:
            LB = mid_s
        
    if best_solution is None:
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    xs = best_solution[0:12]
    ys = best_solution[12:24]
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], 0.0])
        
    return inner, (0.0, 0.0), best_s, 0.0
