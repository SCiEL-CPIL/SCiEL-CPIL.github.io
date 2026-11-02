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
LOGSUMEXP_BETA = 50.0  # Temperature for log-sum-exp smooth max

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
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
        (0, 1), (1, 1), (2, 1), (3, 1),
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

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # Additional candidates: more compact cluster shapes
    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (2, 3)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_38)

    # Additional compact configurations
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, -2)
    ]
    candidates.append(axial_32)

    def axial_to_cartesian(axial_list):
        pts = []
        # Scale by 1.01 to ensure strict disjointness (small gap)
        # Prevents "overlap" errors due to touching edges and floating point precision
        scale = 1.01
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        # For each normal, compute max and min projection
        projs = pts @ normals.T  # shape (11, 6)
        a = np.max(projs, axis=0)  # max projection for each normal
        b = np.min(projs, axis=0)  # min projection for each normal
        
        # Solve LP: minimize D subject to
        # a_k - n_k · c <= D  =>  -n_kx*c_x - n_ky*c_y - D <= -a_k
        # n_k · c - b_k <= D  =>   n_kx*c_x + n_ky*c_y - D <= b_k
        # Variables: [cx, cy, D]
        
        c_A = np.zeros((12, 3))
        b_b = np.zeros(12)
        
        for k in range(6):
            nx, ny = normals[k]
            # -nx*cx - ny*cy - D <= -a_k
            c_A[2*k] = [-nx, -ny, -1.0]
            b_b[2*k] = -a[k]
            # nx*cx + ny*cy - D <= b_k
            c_A[2*k+1] = [nx, ny, -1.0]
            b_b[2*k+1] = b[k]
        
        c_obj = np.array([0.0, 0.0, 1.0])
        
        try:
            res = linprog(c_obj, A_ub=c_A, b_ub=b_b, bounds=[(None, None), (None, None), (0, None)])
            if res.success:
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
            else:
                # Fallback to Nelder-Mead
                def D_of_center(cx, cy):
                    c = np.array([cx, cy])
                    nc = normals @ c
                    vals = np.maximum(a - nc, nc - b)
                    return np.max(vals)
                
                cx_center = (np.max(pts[:, 0]) + np.min(pts[:, 0])) / 2
                cy_center = (np.max(pts[:, 1]) + np.min(pts[:, 1])) / 2
                res = minimize(D_of_center, [cx_center, cy_center], method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            def D_of_center(cx, cy):
                c = np.array([cx, cy])
                nc = normals @ c
                vals = np.maximum(a - nc, nc - b)
                return np.max(vals)
            
            cx_center = (np.max(pts[:, 0]) + np.min(pts[:, 0])) / 2
            cy_center = (np.max(pts[:, 1]) + np.min(pts[:, 1])) / 2
            res = minimize(D_of_center, [cx_center, cy_center], method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            best_D = res.fun
            best_c = (res.x[0], res.x[1])
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, best_c, best_D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta (13 points is sufficient)
        for i in range(13):
            beta = i * (math.pi / 6) / 12
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Final high-precision polish using smooth log-sum-exp surrogate
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def smooth_objective(params):
                    """Differentiable surrogate using log-sum-exp for smooth max."""
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                        for k in range(6)
                    ])
                    projs = (pts - np.array([cx, cy])) @ normals.T  # (11, 6)
                    abs_projs = np.abs(projs)  # (11, 6)
                    
                    # Log-sum-exp over all 66 values to get smooth max
                    # max_k max_i |proj_{k,i}| ≈ (1/beta_lse) * log(sum(exp(beta_lse * x)))
                    flat = abs_projs.flatten()
                    max_flat = np.max(flat)
                    logsumexp = max_flat + np.log(np.sum(np.exp(LOGSUMEXP_BETA * (flat - max_flat)))) / LOGSUMEXP_BETA
                    S = (logsumexp + APOTHEM) / APOTHEM
                    return S
                
                def exact_objective(params):
                    """Exact nonsmooth objective for final evaluation."""
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
                
                # Alternating optimization with smooth surrogate
                cx, cy = best_center
                beta = best_beta
                
                for iteration in range(8):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    # Optimize cx, cy for fixed beta using smooth objective
                    def obj_c_smooth(params):
                        return smooth_objective([params[0], params[1], beta])
                    
                    res_c = minimize(obj_c_smooth, [cx, cy], method='Nelder-Mead',
                                     options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12})
                    cx, cy = res_c.x
                    
                    # Optimize beta for fixed cx, cy using smooth objective
                    def obj_b_smooth(params):
                        return smooth_objective([cx, cy, params[0]])
                    
                    # Multi-start for beta
                    b_starts = np.linspace(0, math.pi/6, 30)
                    best_b = beta
                    best_S_b = exact_objective([cx, cy, beta])
                    for b0 in b_starts:
                        if abs(b0 - best_b) < 5e-4:
                            continue
                        res_b = minimize_scalar(obj_b_smooth, bounds=(0, math.pi/6), method='bounded',
                                                options={'maxiter': 150, 'xatol': 1e-12})
                        S_exact = exact_objective([cx, cy, res_b.x])
                        if S_exact < best_S_b:
                            best_S_b = S_exact
                            best_b = res_b.x
                    beta = best_b
                    
                    # Evaluate with exact objective
                    S_current = exact_objective([cx, cy, beta])
                    if S_current < best_S:
                        best_S = S_current
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
    
    # Compute exact S from the optimized configuration
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    # Final micro-polish: try tiny perturbations of center and angle to shave off any remaining slack
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        best_micro_S = S_final
        best_micro_cx, best_micro_cy = cx, cy
        best_micro_beta = best_global_beta
        
        # Try small grid of center offsets
        for dx in [-0.01, 0, 0.01]:
            for dy in [-0.01, 0, 0.01]:
                for dbeta in [-0.005, 0, 0.005]:
                    test_cx = cx + dx
                    test_cy = cy + dy
                    test_beta = best_global_beta + dbeta
                    mp = 0
                    for k in range(6):
                        ang = test_beta + math.pi / 6.0 + k * math.pi / 3.0
                        nx, ny = math.cos(ang), math.sin(ang)
                        projs = (best_global_pts[:, 0] - test_cx) * nx + (best_global_pts[:, 1] - test_cy) * ny
                        mp = max(mp, np.max(projs), -np.min(projs))
                    test_S = (mp + APOTHEM) / APOTHEM
                    if test_S < best_micro_S:
                        best_micro_S = test_S
                        best_micro_cx, best_micro_cy = test_cx, test_cy
                        best_micro_beta = test_beta
        
        if best_micro_S < S_final:
            cx, cy = best_micro_cx, best_micro_cy
            best_global_beta = best_micro_beta
            S_final = best_micro_S
            angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
