import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """
        Calculate the minimum side length S of the outer hexagon
        for a given set of points and orientation beta.
        Returns (S, center, D) or (inf, None, None).
        """
        if time.time() - START_TIME > TIME_LIMIT:
            return float('inf'), None, None
            
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        # Pre-compute M_i^+ and M_i^-
        # pts shape (N, 2)
        # normals shape (6, 2)
        # projs shape (N, 6)
        projs = pts @ np.array(normals).T
        
        for k in range(6):
            nx, ny = normals[k]
            M_plus = np.max(projs[:, k])
            M_minus = np.max(-projs[:, k]) # max of -proj = -min of proj
            
            # Constraint 1: c . n - A_out <= A_in - M_plus
            A_ub.append([nx, ny, -1.0])
            b_ub.append(APOTHEM - M_plus)
            
            # Constraint 2: -c . n - A_out <= A_in - M_minus
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(APOTHEM - M_minus)
        
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

    def minimize_S_over_beta(pts):
        """
        Minimize S over beta in [0, pi/6).
        """
        if time.time() - START_TIME > TIME_LIMIT:
            return float('inf'), 0.0, None
        
        # Coarse grid
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine
        try:
            res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded', options={'maxiter': 100})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    def local_search(initial_axial):
        """
        Perform hill-climbing on the topology.
        """
        current_axial = set(initial_axial)
        current_S, current_beta, current_center = minimize_S_over_beta(axial_to_cartesian(list(current_axial)))
        best_S = current_S
        best_axial = set(current_axial)
        best_beta = current_beta
        best_center = current_center
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - START_TIME < TIME_LIMIT and no_improve_count < max_no_improve:
            improved = False
            # Generate neighbors
            neighbors = []
            for q, r in current_axial:
                for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    nq, nr = q+dq, r+dr
                    if (nq, nr) not in current_axial:
                        # Create new set
                        new_set = set(current_axial)
                        new_set.remove((q,r))
                        new_set.add((nq,nr))
                        neighbors.append(new_set)
            
            # Evaluate neighbors
            for new_set in neighbors:
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                pts = axial_to_cartesian(list(new_set))
                S, beta, center = minimize_S_over_beta(pts)
                if S < best_S - 1e-6:
                    best_S = S
                    best_axial = new_set
                    best_beta = beta
                    best_center = center
                    current_axial = new_set
                    current_S = S
                    current_beta = beta
                    current_center = center
                    improved = True
                    no_improve_count = 0
                    break # Restart neighbor generation from new best
            
            if not improved:
                no_improve_count += 1
                
        return best_S, best_axial, best_beta, best_center

    # Initial topologies
    # 1. Compact 3x3 + 2
    init1 = [(q, r) for q in range(3) for r in range(3)] + [(3,0), (3,1)]
    # 2. Compact 2x5 + 1
    init2 = [(q, r) for q in range(5) for r in range(2)] + [(0,2)]
    # 3. Ring
    init3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    # 4. Random compact
    init4 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (1,2), (0,2), (1,1)]

    best_global_S = float('inf')
    best_global_axial = None
    best_global_beta = 0.0
    best_global_center = None

    for init in [init1, init2, init3, init4]:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        S, axial, beta, center = local_search(init)
        if S < best_global_S:
            best_global_S = S
            best_global_axial = axial
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None:
        # Fallback
        best_global_axial = init1
        best_global_S, best_global_beta, best_global_center = minimize_S_over_beta(axial_to_cartesian(list(best_global_axial)))

    # Prepare output
    pts = axial_to_cartesian(list(best_global_axial))
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale slightly to ensure no floating point issues
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S for the scaled points to be safe
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
