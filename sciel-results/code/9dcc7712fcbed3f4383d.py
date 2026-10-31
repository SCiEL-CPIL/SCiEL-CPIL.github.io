import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: Return a valid but potentially suboptimal packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Define the compact cluster of 11 hexagons in axial coordinates
    # This shape is a known optimal or near-optimal packing for N=11
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Convert axial to Cartesian coordinates
    pts = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    
    # Function to solve for minimum enclosing hexagon side length S
    # given a rotation angle beta for the outer hexagon
    def solve_lp_for_beta(beta):
        # Normals of the outer hexagon faces
        normals = []
        for k in range(6):
            ang = beta + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, A (apothem)
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # Constraint: nx*(px - cx) + ny*(py - cy) <= A
                # => -nx*cx - ny*cy - A <= -nx*px - ny*py
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # Constraint: -(nx*(px - cx) + ny*(py - cy)) <= A
                # => nx*cx + ny*cy - A <= nx*px + ny*py
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                A_opt = res.x[2]
                cx_opt = res.x[0]
                cy_opt = res.x[1]
                # Side length S = 2 * A / sqrt(3)
                S = 2.0 * A_opt / math.sqrt(3)
                return S, (cx_opt, cy_opt), A_opt
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    # Optimize over rotation beta
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse scan
    for i in range(13):
        beta = i * (math.pi / 6.0) / 12.0
        S, center, _ = solve_lp_for_beta(beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center

    # Fine scan using minimize_scalar
    try:
        def objective(beta):
            S, _, _ = solve_lp_for_beta(beta)
            return S
            
        res = minimize_scalar(objective, bounds=(0, math.pi / 6.0), method='bounded', options={'xatol': 1e-8})
        if res.success and res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            _, best_center, _ = solve_lp_for_beta(best_beta)
    except Exception:
        pass

    # Construct the output
    inner_list = []
    for x, y in pts:
        inner_list.append([float(x), float(y), 0.0])
        
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_side = float(best_S)
    outer_angle = float(best_beta)
    
    return inner_list, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
