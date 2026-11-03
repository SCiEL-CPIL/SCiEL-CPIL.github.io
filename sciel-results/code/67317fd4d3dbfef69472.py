import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    # Focusing on compact polyhexes of size 11
    candidates = []

    # 1. 3x3 block + 2 on top (axial_10 from original)
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])

    # 2. 3x3 block + 1 on top + 1 on side
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (3, 1)
    ])

    # 3. 2x4 block + 3 extra
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (1, 3)
    ])

    # 4. "Plus" shape extended
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])

    # 5. 3x3 + 2 diagonal
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (2, 3), (3, 2)
    ])

    # 6. 3x3 + 2 side
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 1), (3, 2)
    ])

    # 7. 4x3 block missing 1
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])

    # 8. 5x2 block + 1
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2)
    ])

    # 9. 3x3 + 2 bottom
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, -1)
    ])

    # 10. 3x3 + 2 top-left
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_lp(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def calc_S_nm(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        projs = pts @ normals.T
        a = np.max(projs, axis=0)
        b = np.min(projs, axis=0)
        
        def D_of_center(cx, cy):
            c = np.array([cx, cy])
            nc = normals @ c
            vals = np.maximum(a - nc, nc - b)
            return np.max(vals)
        
        cx_center = (np.max(pts[:, 0]) + np.min(pts[:, 0])) / 2
        cy_center = (np.max(pts[:, 1]) + np.min(pts[:, 1])) / 2
        
        best_D = float('inf')
        best_c = (cx_center, cy_center)
        
        cx_range = np.max(pts[:, 0]) - np.min(pts[:, 0])
        cy_range = np.max(pts[:, 1]) - np.min(pts[:, 1])
        
        for dx in np.linspace(-cx_range*0.3, cx_range*0.3, 5):
            for dy in np.linspace(-cy_range*0.3, cy_range*0.3, 5):
                D = D_of_center(cx_center + dx, cy_center + dy)
                if D < best_D:
                    best_D = D
                    best_c = (cx_center + dx, cy_center + dy)
        
        try:
            res = minimize(D_of_center, best_c, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_D:
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            pass
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, best_c, best_D

    def calc_S(pts, beta):
        S_lp, c_lp, D_lp = calc_S_lp(pts, beta)
        if S_lp < float('inf'):
            return S_lp, c_lp, D_lp
        return calc_S_nm(pts, beta)

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        for i in range(13):
            beta = i * (math.pi / 6) / 12
            S, center, D = calc_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S(pts, best_beta)
        except Exception:
            pass
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                        for k in range(6)
                    ])
                    projs = (pts - np.array([cx, cy])) @ normals.T
                    max_proj = np.max(np.abs(projs))
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                cx, cy = best_center
                beta = best_beta
                
                for _ in range(5):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    def obj_c(params):
                        return objective([params[0], params[1], beta])
                    
                    res_c = minimize(obj_c, [cx, cy], method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                    cx, cy = res_c.x
                    S_c = res_c.fun
                    
                    def obj_b(params):
                        return objective([cx, cy, params[0]])
                    
                    b_starts = np.linspace(0, math.pi/6, 20)
                    best_b = beta
                    best_S_b = S_c
                    for b0 in b_starts:
                        if abs(b0 - best_b) < 1e-3: continue
                        res_b = minimize_scalar(obj_b, bounds=(0, math.pi/6), method='bounded',
                                                options={'maxiter': 100, 'xatol': 1e-10})
                        if res_b.fun < best_S_b:
                            best_S_b = res_b.fun
                            best_b = res_b.x
                    beta = best_b
                    S_b = best_S_b
                    
                    if S_b < best_S:
                        best_S = S_b
                        best_center = (cx, cy)
                        best_beta = beta
                        
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
