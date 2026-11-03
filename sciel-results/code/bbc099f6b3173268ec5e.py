import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 110.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - 5.0:
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts.append([x, y, 0.0])
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        max_proj = 0
        for k in range(6):
            ang = math.pi/6 + k*math.pi/3
            nx, ny = math.cos(ang), math.sin(ang)
            for x, y, _ in pts:
                proj = abs((x-cx)*nx + (y-cy)*ny)
                max_proj = max(max_proj, proj)
        S = (max_proj + APOTHEM) / APOTHEM
        return pts, [cx, cy], float(S), 0.0

    def get_all_vertices(pts, angles_rad):
        """Get all 6N vertices of N unit hexagons."""
        verts = np.zeros((6*N, 2))
        for i in range(N):
            cx_i, cy_i = pts[i]
            a = angles_rad[i]
            for k in range(6):
                theta = a + k * math.pi / 3.0
                verts[6*i + k, 0] = cx_i + math.cos(theta)
                verts[6*i + k, 1] = cy_i + math.sin(theta)
        return verts

    def enclosing_hexagon_for_beta(verts, beta):
        """For fixed outer orientation beta, find min D and center via LP."""
        normals = np.array([(math.cos(beta + k*math.pi/3), math.sin(beta + k*math.pi/3)) for k in range(6)])
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for px, py in verts:
            for nx, ny in normals:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px*nx + py*ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px*nx + py*ny)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[2], res.x[0], res.x[1]
        except:
            pass
        return float('inf'), 0.0, 0.0

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - 2.0:
            return 1e10
        pts = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        verts = get_all_vertices(pts, angles)
        
        # Optimize over beta
        best_S = float('inf')
        # Coarse scan
        for i in range(6):
            beta = i * (math.pi/6) / 6
            D, cx, cy = enclosing_hexagon_for_beta(verts, beta)
            if D / APOTHEM < best_S:
                best_S = D / APOTHEM
        # Refine
        try:
            res = minimize_scalar(lambda b: enclosing_hexagon_for_beta(verts, b)[0] / APOTHEM,
                                 bounds=(0, math.pi/6), method='bounded', options={'maxiter': 30})
            if res.fun < best_S:
                best_S = res.fun
        except:
            pass
        return best_S

    best_S = float('inf')
    best_params = None
    rng = np.random.default_rng(123)

    axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]
    base_pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        base_pts.append([x, y])
    base_pts = np.array(base_pts)

    n_restarts = 25
    for r in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 8.0:
            break
        if r == 0:
            pts = base_pts.copy()
            angles = np.zeros(N)
        else:
            spread = 0.1 + rng.random() * 1.0
            pts = base_pts + rng.normal(0, spread, size=(N, 2))
            angles = rng.uniform(0, math.pi/6, size=N)
        params = np.concatenate([pts.ravel(), angles])
        
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    if best_params is None:
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts.append([x, y, 0.0])
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        max_proj = 0
        for k in range(6):
            ang = math.pi/6 + k*math.pi/3
            nx, ny = math.cos(ang), math.sin(ang)
            for x, y, _ in pts:
                proj = abs((x-cx)*nx + (y-cy)*ny)
                max_proj = max(max_proj, proj)
        S = (max_proj + APOTHEM) / APOTHEM
        return pts, [cx, cy], float(S), 0.0

    # Final polish
    if time.time() - START_TIME < TIME_LIMIT - 5.0:
        try:
            res_f = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11})
            if res_f.fun < best_S:
                best_S = res_f.fun
                best_params = res_f.x.copy()
        except:
            pass

    pts = best_params[:2*N].reshape(N, 2)
    angles_rad = best_params[2*N:]
    angles_deg = np.degrees(angles_rad)
    verts = get_all_vertices(pts, angles_rad)
    
    # Find best beta
    best_D = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_beta = 0.0
    for i in range(12):
        beta = i * (math.pi/6) / 12
        D, cx, cy = enclosing_hexagon_for_beta(verts, beta)
        if D < best_D:
            best_D = D
            best_cx = cx
            best_cy = cy
            best_beta = beta
    try:
        res_b = minimize_scalar(lambda b: enclosing_hexagon_for_beta(verts, b)[0],
                               bounds=(0, math.pi/6), method='bounded', options={'maxiter': 50})
        if res_b.fun < best_D:
            best_D = res_b.fun
            best_beta = res_b.x
            _, best_cx, best_cy = enclosing_hexagon_for_beta(verts, best_beta)
    except:
        pass

    S = best_D / APOTHEM
    inner = []
    for i in range(N):
        inner.append([float(pts[i, 0]), float(pts[i, 1]), float(angles_deg[i])])
    return inner, [float(best_cx), float(best_cy)], float(S), float(math.degrees(best_beta))
