import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def hex_support(alpha, direction_angle):
    """
    Compute the support function of a unit regular hexagon with rotation alpha
    in the given direction.
    For a regular hexagon of side 1 (circumradius 1), vertices are at angles
    alpha + k*pi/3 for k=0..5.
    Support in direction theta = max_k cos(theta - alpha - k*pi/3)
    """
    diff = (direction_angle - alpha) % (math.pi / 3.0)
    d = min(diff, math.pi / 3.0 - diff)
    return math.cos(d)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Multiple candidate configurations: different topologies of 11 hexagons
    # Each is a list of axial (q, r) coordinates
    candidate_axials = [
        # Config 0: center + 6 neighbors + 4 from ring 2
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ],
        # Config 1: center + 6 neighbors + 4 alternating from ring 2
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (0, 2), (-2, 2), (2, -2)
        ],
        # Config 2: center + 6 neighbors + 4 clustered
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, -2), (-1, 2), (-2, 1)
        ],
        # Config 3: center + 6 neighbors + 4 spread
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (0, 2), (-2, 2), (2, -2)
        ],
        # Config 4: center + 6 neighbors + 4 diagonal
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (-1, 2), (-2, 0), (1, -2)
        ],
    ]
    
    SCALE = 1.0 + 1e-6
    
    def axial_to_points(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0) * SCALE
            y = 1.5 * r * SCALE
            pts.append((x, y))
        return np.array(pts)
    
    def solve_lp(beta, inner_angles, pts_arr, time_limit=TIME_LIMIT - SAFETY_MARGIN):
        """
        Solves the LP for a given outer rotation beta and inner hexagon rotations.
        Returns (cx, cy, S) or (None, None, inf)
        """
        if time.time() - START_TIME > time_limit:
            return None, None, float('inf')
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for i, (px, py) in enumerate(pts_arr):
            alpha_i = inner_angles[i]
            for nx, ny in normals:
                dir_ang = math.atan2(ny, nx)
                h_i = hex_support(alpha_i, dir_ang)
                
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny + h_i))
                
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny + h_i)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
                S = D / APOTHEM
                return cx, cy, S
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')

    def smooth_solve_lp(beta, inner_angles, pts_arr, temperature=10.0):
        """
        Smooth approximation of the LP using log-sum-exp.
        Returns (cx, cy, S_smooth) where S is an estimate of the optimal value.
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        def objective(x):
            cx, cy = x[0], x[1]
            
            # Compute all constraint values: (p_i - c).n_k + h_i(n_k) and -(p_i-c).n_k + h_i(n_k)
            constraints = []
            for i, (px, py) in enumerate(pts_arr):
                alpha_i = inner_angles[i]
                for nx, ny in normals:
                    dir_ang = math.atan2(ny, nx)
                    h_i = hex_support(alpha_i, dir_ang)
                    v1 = (px - cx) * nx + (py - cy) * ny + h_i
                    v2 = -(px - cx) * nx - (py - cy) * ny + h_i
                    constraints.append(v1)
                    constraints.append(v2)
            
            constraints = np.array(constraints)
            max_c = np.max(constraints)
            
            # Smooth max via log-sum-exp
            shifted = constraints - max_c
            shifted = np.clip(shifted, -50, 50)
            smooth_max = max_c + (1.0 / temperature) * math.log(np.sum(np.exp(temperature * shifted)))
            
            S = smooth_max / APOTHEM
            return S
        
        cx0, cy0 = 0.0, 0.0
        x0 = [cx0, cy0]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            cx, cy = res.x
            S = res.fun
            return cx, cy, S
        except Exception:
            return None, None, float('inf')

    # Best solution across all configurations
    global_best_S = float('inf')
    global_best_cx = 0.0
    global_best_cy = 0.0
    global_best_beta = 0.0
    global_best_angles = [0.0] * N
    global_best_pts = None
    
    for config_idx, axial_list in enumerate(candidate_axials):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_points(axial_list)
        n_pts = len(pts)
        
        # Start with all inner hexagons aligned with outer
        def obj_aligned(beta, pts=pts, n_pts=n_pts):
            inner_angles = [beta] * n_pts
            _, _, S = solve_lp(beta, inner_angles, pts)
            return S

        best_beta = 0.0
        best_cx, best_cy, best_S = solve_lp(best_beta, [best_beta] * n_pts, pts)
        
        if best_S == float('inf'):
            continue

        # Minimize over beta in [0, pi/6]
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(obj_aligned, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_beta = res.x
                    best_cx, best_cy, best_S = solve_lp(best_beta, [best_beta] * n_pts, pts)
        except Exception:
            pass

        # Coordinate descent on inner angles
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            inner_angles = [best_beta] * n_pts
            
            for pass_num in range(3):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                for i in range(n_pts):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    def obj_angle_i(theta_i, i=i):
                        angles = inner_angles[:]
                        angles[i] = theta_i
                        _, _, S = solve_lp(best_beta, angles, pts)
                        return S
                    
                    current_angle = inner_angles[i]
                    best_angle_i = current_angle
                    best_S_i = obj_angle_i(current_angle)
                    
                    candidates = list(np.linspace(current_angle - math.pi/6, current_angle + math.pi/6, 7))
                    for base in [0, math.pi/6, math.pi/3]:
                        candidates.append(base)
                    
                    for cand in candidates:
                        S_cand = obj_angle_i(cand)
                        if S_cand < best_S_i:
                            best_S_i = S_cand
                            best_angle_i = cand
                    
                    if best_S_i < best_S - 1e-12:
                        inner_angles[i] = best_angle_i
                        best_S = best_S_i
                        best_cx, best_cy, _ = solve_lp(best_beta, inner_angles, pts)
        
        # Final beta polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_beta_final(beta):
                _, _, S = solve_lp(beta, inner_angles, pts)
                return S
            
            try:
                res = minimize_scalar(obj_beta_final, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_beta = res.x
                    best_cx, best_cy, best_S = solve_lp(best_beta, inner_angles, pts)
            except Exception:
                pass
        
        # Update global best
        if best_S < global_best_S:
            global_best_S = best_S
            global_best_cx = best_cx
            global_best_cy = best_cy
            global_best_beta = best_beta
            global_best_angles = inner_angles[:]
            global_best_pts = pts

    # Final check
    if global_best_S == float('inf') or global_best_pts is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # High-precision polish: jointly optimize (cx, cy, beta, inner_angles)
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        pts = global_best_pts
        n_pts = len(pts)
        
        def smooth_objective(x):
            cx, cy = x[0], x[1]
            beta = x[2]
            angles = x[3:]
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            constraints = []
            for i, (px, py) in enumerate(pts):
                alpha_i = angles[i]
                for nx, ny in normals:
                    dir_ang = math.atan2(ny, nx)
                    h_i = hex_support(alpha_i, dir_ang)
                    v1 = (px - cx) * nx + (py - cy) * ny + h_i
                    v2 = -(px - cx) * nx - (py - cy) * ny + h_i
                    constraints.append(v1)
                    constraints.append(v2)
            
            constraints = np.array(constraints)
            max_c = np.max(constraints)
            
            temp = 100.0
            shifted = constraints - max_c
            shifted = np.clip(shifted, -50, 50)
            smooth_max = max_c + (1.0 / temp) * math.log(np.sum(np.exp(temp * shifted)))
            
            S = smooth_max / APOTHEM
            return S
        
        x0 = np.array([global_best_cx, global_best_cy, global_best_beta] + list(global_best_angles))
        
        try:
            res = minimize(smooth_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            
            if res.fun < global_best_S:
                best_cx, best_cy = float(res.x[0]), float(res.x[1])
                best_beta = float(res.x[2])
                best_angles = [float(a) for a in res.x[3:]]
                _, _, S_exact = solve_lp(best_beta, best_angles, pts)
                if S_exact < global_best_S:
                    global_best_S = S_exact
                    global_best_cx = best_cx
                    global_best_cy = best_cy
                    global_best_beta = best_beta
                    global_best_angles = best_angles
        except Exception:
            pass

    # Prepare output
    inner = []
    for i, (x, y) in enumerate(global_best_pts):
        angle_deg = math.degrees(global_best_angles[i])
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(global_best_cx), float(global_best_cy)]
    side = float(global_best_S)
    rotation = math.degrees(global_best_beta)
    
    return inner, center, side, rotation
