import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define several lattice configurations
    # 1. Current "Asymmetric Flower"
    config_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # 2. Symmetric "3-5-3" Lattice
    config_2 = [
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1)
    ]
    
    configs = [config_1, config_2]
    
    best_S = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_beta = 0.0
    best_pts = None

    for cfg in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        SCALE = 1.0 + 1e-6
        pts = []
        for q, r in cfg:
            x = math.sqrt(3) * (q + r / 2.0) * SCALE
            y = 1.5 * r * SCALE
            pts.append((x, y))
        pts = np.array(pts)
        n_pts = len(pts)
        
        def solve_for_beta(beta, pts, cx_init=0.0, cy_init=0.0):
            """
            Solves the full LP for a given outer rotation beta and fixed inner points.
            Returns (cx, cy, S) or (None, None, inf)
            """
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return None, None, float('inf')
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            c = np.array([0.0, 0.0, 1.0])
            
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
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    cx = res.x[0]
                    cy = res.x[1]
                    D = res.x[2]
                    S = (D + APOTHEM) / APOTHEM
                    return cx, cy, S
                else:
                    return None, None, float('inf')
            except Exception:
                return None, None, float('inf')

        def obj(beta):
            _, _, S = solve_for_beta(beta, pts)
            return S

        # Initial guess
        cx0, cy0, S0 = solve_for_beta(0.0, pts)
        if S0 == float('inf'):
            continue
            
        best_beta_local = 0.0
        best_S_local = S0
        best_cx_local = cx0
        best_cy_local = cy0

        # Optimize beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Coarse search
                res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-5})
                if res.fun < best_S_local:
                    best_beta_local = res.x
                    cx, cy, S = solve_for_beta(best_beta_local, pts)
                    if S < best_S_local:
                        best_S_local = S
                        best_cx_local = cx
                        best_cy_local = cy
                        
                    # Fine search
                    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                        local_bounds = (max(0, best_beta_local - 0.01), min(math.pi / 6, best_beta_local + 0.01))
                        res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                              options={'maxiter': 200, 'xatol': 1e-8})
                        if res2.fun < best_S_local:
                            best_beta_local = res2.x
                            cx, cy, S = solve_for_beta(best_beta_local, pts)
                            if S < best_S_local:
                                best_S_local = S
                                best_cx_local = cx
                                best_cy_local = cy
                                
                                # Ultra fine
                                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                                    local_bounds2 = (max(0, best_beta_local - 1e-4), min(math.pi / 6, best_beta_local + 1e-4))
                                    res3 = minimize_scalar(obj, bounds=local_bounds2, method='bounded',
                                                          options={'maxiter': 500, 'xatol': 1e-12})
                                    if res3.fun < best_S_local:
                                        best_beta_local = res3.x
                                        cx, cy, S = solve_for_beta(best_beta_local, pts)
                                        if S < best_S_local:
                                            best_S_local = S
                                            best_cx_local = cx
                                            best_cy_local = cy
        except Exception:
            pass
        
        if best_S_local < best_S:
            best_S = best_S_local
            best_cx = best_cx_local
            best_cy = best_cy_local
            best_beta = best_beta_local
            best_pts = pts

    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
