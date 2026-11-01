import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate axial point sets
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """Solve LP for given outer rotation. Returns (S, (cx,cy), D) or (inf, None, None)."""
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

    def optimize_shape(pts):
        """Optimize over beta for a fixed point set using coarse scan + local polish."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse scan
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Local polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj(beta):
                S, _, _ = calc_S_for_beta(pts, beta)
                return S
            
            local_bounds = (max(0, best_beta - 0.02), min(math.pi / 6, best_beta + 0.02))
            try:
                res = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_beta = res.x
                    S, center, D = calc_S_for_beta(pts, best_beta)
                    best_S = S
                    best_center = center
            except Exception:
                pass
        
        return best_S, best_beta, best_center

    # Evaluate all candidates
    best_overall_S = float('inf')
    best_overall_pts = None
    best_overall_beta = 0.0
    best_overall_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        
        if S < best_overall_S:
            best_overall_S = S
            best_overall_pts = pts
            best_overall_beta = beta
            best_overall_center = center

    # If all failed, return fallback
    if best_overall_S == float('inf') or best_overall_pts is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Phase 2: Joint optimization using smooth surrogate
    # Variables: [cx, cy, beta, S]
    cx0, cy0 = best_overall_center
    x0 = np.array([cx0, cy0, best_overall_beta, best_overall_S])
    pts = best_overall_pts
    n_pts = len(pts)

    def smooth_objective(x):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        cx, cy, beta, S = x
        S = max(S, 0.1)
        
        normals = np.array([[math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                            math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
        
        # Compute all margins
        all_margins = np.zeros((n_pts, 12))
        idx_m = 0
        for i in range(n_pts):
            for k in range(6):
                all_margins[i, idx_m] = S * APOTHEM - APOTHEM - ((pts[i, 0] - cx) * normals[k, 0] + (pts[i, 1] - cy) * normals[k, 1])
                idx_m += 1
            for k in range(6):
                all_margins[i, idx_m] = S * APOTHEM - APOTHEM - (-(pts[i, 0] - cx) * normals[k, 0] - (pts[i, 1] - cy) * normals[k, 1])
                idx_m += 1
        
        all_margins_flat = all_margins.flatten()
        tau = 0.05
        neg_margins = -all_margins_flat
        max_neg = np.max(neg_margins)
        if max_neg > 100:
            lse = max_neg + np.log(np.sum(np.exp(neg_margins - max_neg)))
        else:
            lse = np.log(np.sum(np.exp(neg_margins)))
        soft_min = -lse / tau
        
        penalty = max(0.0, soft_min)
        obj = S + 10.0 * penalty
        
        return obj

    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res_smooth = minimize(smooth_objective, x0, method='L-BFGS-B',
                                  bounds=[(-2, 2), (-2, 2), (0, math.pi/6), (0.5, 5.0)],
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res_smooth.fun < best_overall_S and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_new, cy_new, beta_new, S_new = res_smooth.x
                # Verify with exact LP
                S_lp, center_lp, D_lp = calc_S_for_beta(pts, beta_new)
                if S_lp < best_overall_S:
                    best_overall_S = S_lp
                    best_overall_beta = beta_new
                    best_overall_center = center_lp
    except Exception:
        pass

    # Phase 3: Final high-precision LP polish
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_final(beta):
                S, _, _ = calc_S_for_beta(pts, beta)
                return S
            
            local_bounds = (max(0, best_overall_beta - 0.005), min(math.pi / 6, best_overall_beta + 0.005))
            res3 = minimize_scalar(obj_final, bounds=local_bounds, method='bounded',
                                  options={'maxiter': 1000, 'xatol': 1e-12})
            if res3.fun < best_overall_S:
                best_overall_beta = res3.x
                S_lp, center_lp, D_lp = calc_S_for_beta(pts, best_overall_beta)
                if S_lp < best_overall_S:
                    best_overall_S = S_lp
                    best_overall_center = center_lp
                    
                # Even tighter
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds2 = (max(0, best_overall_beta - 0.001), min(math.pi / 6, best_overall_beta + 0.001))
                    res4 = minimize_scalar(obj_final, bounds=local_bounds2, method='bounded',
                                          options={'maxiter': 2000, 'xatol': 1e-14})
                    if res4.fun < best_overall_S:
                        best_overall_beta = res4.x
                        S_lp, center_lp, D_lp = calc_S_for_beta(pts, best_overall_beta)
                        if S_lp < best_overall_S:
                            best_overall_S = S_lp
                            best_overall_center = center_lp
    except Exception:
        pass

    # Final check
    if best_overall_S == float('inf') or best_overall_center is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Prepare output
    inner = []
    angle_deg = math.degrees(best_overall_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_overall_center[0]), float(best_overall_center[1])]
    side = float(best_overall_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
