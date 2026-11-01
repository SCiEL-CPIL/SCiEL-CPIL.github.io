import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Base 7 hexagons (1 center + 6 ring 1)
    base_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # Candidate sets for the 4 ring-2 hexagons
    # Axial coordinates for ring 2
    candidates = []
    
    # C1: Current program's selection
    c1_extra = [(1, 1), (-1, 2), (-2, 1), (2, -1)]
    candidates.append(base_axial + c1_extra)
    
    # C2a: Opposite pairs (2,0)-(-2,0) and (1,1)-(-1,-1)
    c2a_extra = [(2, 0), (-2, 0), (1, 1), (-1, -1)]
    candidates.append(base_axial + c2a_extra)
    
    # C2b: Opposite pairs (2,0)-(-2,0) and (0,2)-(0,-2)
    c2b_extra = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    candidates.append(base_axial + c2b_extra)
    
    # C2c: Opposite pairs (1,1)-(-1,-1) and (0,2)-(0,-2)
    c2c_extra = [(1, 1), (-1, -1), (0, 2), (0, -2)]
    candidates.append(base_axial + c2c_extra)
    
    # C2d: Opposite pairs (-1,2)-(1,-2) and (-2,1)-(2,-1)
    c2d_extra = [(-1, 2), (1, -2), (-2, 1), (2, -1)]
    candidates.append(base_axial + c2d_extra)
    
    # C3: 3-fold symmetric? Not possible with 4.
    # C4: "L" shape or compact block
    # (1,1), (0,2), (-1,2), (-2,1) - Block in NW
    c4_extra = [(1, 1), (0, 2), (-1, 2), (-2, 1)]
    candidates.append(base_axial + c4_extra)

    SCALE = 1.0 + 1e-6
    
    best_S = float('inf')
    best_config = None
    best_beta = 0.0
    best_center = None
    best_axial = None

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None
        
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
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    for axial_set in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = []
        for q, r in axial_set:
            x = math.sqrt(3) * (q + r / 2.0) * SCALE
            y = 1.5 * r * SCALE
            pts.append((x, y))
        pts = np.array(pts)
        
        # Coarse search for beta
        local_best_S = float('inf')
        local_best_beta = 0.0
        local_best_center = None
        
        for i in range(31): # 0 to 30 deg
            beta = i * (math.pi / 6) / 30.0
            S, center = calc_S_for_beta(pts, beta)
            if S < local_best_S:
                local_best_S = S
                local_best_beta = beta
                local_best_center = center
                
        # Refine
        if local_best_S < float('inf'):
            try:
                # Bound the refinement around the best grid point
                # The function is periodic with pi/6.
                # We can just minimize in [0, pi/6]
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 50, 'xatol': 1e-8})
                if res.fun < local_best_S:
                    local_best_S = res.fun
                    local_best_beta = res.x
                    _, local_best_center = calc_S_for_beta(pts, local_best_beta)
            except Exception:
                pass

        if local_best_S < best_S:
            best_S = local_best_S
            best_beta = local_best_beta
            best_center = local_best_center
            best_axial = axial_set

    if best_axial is None or best_S == float('inf'):
        # Fallback to current program's logic
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Prepare output
    angle_deg = math.degrees(best_beta)
    inner = []
    for q, r in best_axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
