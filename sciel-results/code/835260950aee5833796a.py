import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave a safety margin for the evaluator

def construct():
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate configurations (axial coordinates)
    # Each configuration is a list of (q, r)
    candidates = []
    
    # 1. The "flower" from the current program (Center + Ring 1 + 4 from Ring 2)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ])
    
    # 2. "Plus" shape with corners (Center + Ring 1 + 4 cardinal from Ring 2)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    
    # 3. A 3x3 block minus 2 corners
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ])
    
    # 4. A 3x4 rectangle minus 1 corner
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 5. Center + Ring 1 + 3 from Ring 2 (symmetric subset)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2)
    ])
    
    # 6. Another compact shape
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def solve_lp_for_beta(pts, beta):
        """
        Solves the LP for a given outer rotation beta.
        Returns (S, cx, cy) or (inf, None, None)
        """
        if time.time() - START_TIME > TIME_LIMIT:
            return float('inf'), None, None
            
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D
        c = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)nx + (py - cy)ny <= D
                # px*nx - cx*nx + py*ny - cy*ny - D <= 0
                # -nx*cx - ny*cy - D <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -((px - cx)nx + (py - cy)ny) <= D
                # (cx - px)nx + (cy - py)ny - D <= 0
                # nx*cx + ny*cy - D <= px*nx + py*ny
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
                return S, cx, cy
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    best_S = float('inf')
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    best_pts = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Coarse search for beta
        coarse_betas = np.linspace(0, math.pi / 6, 37)
        best_local_S = float('inf')
        best_local_beta = 0.0
        best_local_cx = 0.0
        best_local_cy = 0.0
        
        for beta in coarse_betas:
            S, cx, cy = solve_lp_for_beta(pts, beta)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = beta
                best_local_cx = cx
                best_local_cy = cy
        
        if best_local_S >= best_S:
            continue
            
        # Fine search around the best coarse beta
        try:
            if time.time() - START_TIME < TIME_LIMIT:
                bounds = (max(0, best_local_beta - math.pi/18), min(math.pi/6, best_local_beta + math.pi/18))
                res = minimize_scalar(lambda b: solve_lp_for_beta(pts, b)[0], bounds=bounds, method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_beta = res.x
                    _, best_local_cx, best_local_cy = solve_lp_for_beta(pts, best_local_beta)
        except Exception:
            pass

        if best_local_S < best_S:
            best_S = best_local_S
            best_beta = best_local_beta
            best_cx = best_local_cx
            best_cy = best_local_cy
            best_pts = pts

    # If no valid solution found, return fallback
    if best_S == float('inf') or best_pts is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Final high-precision polish
    try:
        if time.time() - START_TIME < TIME_LIMIT:
            bounds = (max(0, best_beta - 0.01), min(math.pi/6, best_beta + 0.01))
            res = minimize_scalar(lambda b: solve_lp_for_beta(best_pts, b)[0], bounds=bounds, method='bounded',
                                 options={'maxiter': 200, 'xatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_cx, best_cy = solve_lp_for_beta(best_pts, best_beta)
    except Exception:
        pass

    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg

    return inner, center, side, rotation
