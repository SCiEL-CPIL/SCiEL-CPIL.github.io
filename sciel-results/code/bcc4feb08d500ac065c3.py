import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagon_vertices(cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        verts = []
        for k in range(6):
            angle = rot + k * math.pi / 3.0
            x = cx + side * math.cos(angle)
            y = cy + side * math.sin(angle)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        dx = px - cx
        dy = py - cy
        dx_local = dx * math.cos(rot) + dy * math.sin(rot)
        dy_local = -dx * math.sin(rot) + dy * math.cos(rot)
        apothem = side * APOTHEM
        for k in range(6):
            angle = k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            dot = dx_local * nx + dy_local * ny
            if dot > apothem + 1e-10:
                return False
        return True

    def hexagons_disjoint(c1, c2, s1, s2, rot1_deg, rot2_deg):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist >= (s1 + s2) - 1e-9

    # Generate candidate configurations
    candidates = []

    # Standard hexagonal cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Dense 4-wide
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])

    # Corner-filling templates
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(0,1),(0,-1),(1,1)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(1,1),(-1,1),(0,1)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)])
    candidates.append([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)])

    def calc_S_for_beta(pts, beta):
        """Vectorized LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
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

    def optimize_orientation_only(pts):
        """1D scalar minimization over outer hexagon orientation beta in [0, pi/6]."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        n_grid = 97
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1)
                lo = max(0.0, best_beta - 2 * window)
                hi = min(math.pi / 6, best_beta + 2 * window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1) * 0.5
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        """Hybrid optimization with continuous refinement."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        S1, beta1, center1 = optimize_orientation_only(pts)
        if center1 is not None:
            best_S = S1
            best_beta = beta1
            best_center = center1

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.05, 0.02, 0.01]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.005],
                    [best_center[0], best_center[1], best_beta - 0.005],
                    [best_center[0] + 0.005, best_center[1], best_beta],
                    [best_center[0] - 0.005, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.005, best_beta],
                    [best_center[0], best_center[1] - 0.005, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Continuous position optimization via smooth surrogate
    def continuous_optimize(pts_init, max_iter=50):
        """
        Optimize continuous positions of hexagon centers using smooth surrogate.
        Uses log-sum-exp for smooth max and collision penalty.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return pts_init, float('inf')
        
        pts = pts_init.copy()
        n = len(pts)
        
        # Initial center and rotation
        cx_init = float(np.mean(pts[:, 0]))
        cy_init = float(np.mean(pts[:, 1]))
        S_init, beta_init, center_init = optimize_orientation_only(pts)
        
        if center_init is None:
            return pts, float('inf')
        
        best_S = S_init
        best_pts = pts.copy()
        
        # Smooth objective combining container size and collision penalty
        def smooth_objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
            
            # Unpack: positions (2n) + center (2) + rotation (1)
            pos = params[:2*n].reshape(n, 2)
            cx = params[2*n]
            cy = params[2*n+1]
            beta = params[2*n+2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            # Container size (smooth)
            S_container = smooth_calc_S(pos, cx, cy, beta, temperature=0.03)
            
            # Collision penalty (smooth)
            min_dist = 2.0 * APOTHEM
            collision_pen = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist:
                        # Smooth penalty: grows quadratically when overlapping
                        overlap = min_dist - d
                        collision_pen += overlap * overlap * 100.0
            
            return S_container + collision_pen
        
        # Initial parameters
        init_params = np.concatenate([pts.flatten(), [center_init[0], center_init[1], beta_init]])
        
        # Nelder-Mead optimization
        try:
            res = minimize(smooth_objective, init_params, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < best_S + 1e-6:
                pos_opt = res.x[:2*n].reshape(n, 2)
                cx_opt = res.x[2*n]
                cy_opt = res.x[2*n+1]
                beta_opt = ((res.x[2*n+2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                # Check for collisions
                has_collision = False
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                        if d < min_dist - 1e-6:
                            has_collision = True
                            break
                    if has_collision:
                        break
                
                if not has_collision:
                    S_exact = exact_S(pos_opt, cx_opt, cy_opt, beta_opt)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_pts = pos_opt
        except Exception:
            pass
        
        return best_pts, best_S

    # Evaluate all candidates with both grid and continuous optimization
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    # First pass: grid-based optimization
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
    
    # Second pass: try continuous optimization on top candidates
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Try continuous optimization on the best grid solution
        pts_cont, S_cont = continuous_optimize(best_global_pts, max_iter=80)
        if S_cont < best_global_S:
            best_global_S = S_cont
            best_global_pts = pts_cont
            # Recompute center and beta for the continuous solution
            _, best_global_beta, best_global_center = optimize_orientation_only(pts_cont)
    
    # Try continuous optimization on a few other diverse candidates
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        diverse_indices = [0, 6, 14, 30, 35]  # Star, cross, dense, corner-fill, etc.
        for idx in diverse_indices:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            if idx >= len(candidates):
                continue
            pts = axial_to_cartesian(candidates[idx])
            pts_cont, S_cont = continuous_optimize(pts, max_iter=40)
            if S_cont < best_global_S:
                best_global_S = S_cont
                best_global_pts = pts_cont
                _, best_global_beta, best_global_center = optimize_orientation_only(pts_cont)
    
    # Final high-precision polish on the best solution
    if best_global_pts is not None and best_global_center is not None and \
       time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        
        def final_polish_obj(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            return exact_S(best_global_pts, cx, cy, beta)
        
        polish_starts = [
            [best_global_center[0], best_global_center[1], best_global_beta],
            [best_global_center[0], best_global_center[1], best_global_beta + 0.001],
            [best_global_center[0], best_global_center[1], best_global_beta - 0.001],
            [best_global_center[0] + 0.001, best_global_center[1], best_global_beta],
            [best_global_center[0] - 0.001, best_global_center[1], best_global_beta],
            [best_global_center[0], best_global_center[1] + 0.001, best_global_beta],
            [best_global_center[0], best_global_center[1] - 0.001, best_global_beta],
        ]
        
        for init in polish_starts:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(final_polish_obj, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_global_S:
                    best_global_S = res.fun
                    best_global_center = (res.x[0], res.x[1])
                    best_global_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            except Exception:
                pass
    
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

    # Scale points slightly to ensure they fit
    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

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
