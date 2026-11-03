import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

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

    # New candidates
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

    # More compact configurations
    axial_21 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (-1, 2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, 2), (-1, 3)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_30)

    # Additional candidates
    axial_31 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_35)

    # Additional promising configurations
    axial_36 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_40)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                proj = px * nx + py * ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-proj)
                A_ub.append([nx, ny, -1.0])
                b_ub.append(proj)
        
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

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def exact_max_proj(pts, beta, cx, cy):
        """Compute exact max projection (non-smooth) for given center and beta."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        return max_proj

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Better initial center: use LP to find optimal center at beta=0
        S_init, c_init, _ = calc_S_for_beta(pts, 0.0)
        if c_init is not None:
            cx0, cy0 = c_init
        else:
            cx0 = float(np.mean(pts[:, 0]))
            cy0 = float(np.mean(pts[:, 1]))
        
        # Quick grid over beta using smooth surrogate at LP-optimized center
        best_smooth_S = float('inf')
        best_smooth_beta = 0.0
        for i in range(121):  # Even finer grid: 121 points over pi/6
            beta = i * (math.pi / 6) / 120
            S_s, _, _ = smooth_calc_S(pts, beta, cx0, cy0, temperature=0.1)
            if S_s < best_smooth_S:
                best_smooth_S = S_s
                best_smooth_beta = beta
        
        # Now do proper optimization with smooth surrogate from multiple starts
        # This avoids expensive LP calls during search
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
                # More diverse starting betas including neighbors of best
                start_betas = [best_smooth_beta, 0.0, math.pi/12, math.pi/6,
                              best_smooth_beta + 0.02, best_smooth_beta - 0.02,
                              best_smooth_beta + 0.04, best_smooth_beta - 0.04,
                              best_smooth_beta + 0.08, best_smooth_beta - 0.08,
                              math.pi/24, 5*math.pi/24, math.pi/8, 3*math.pi/8]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [cx0, cy0, sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            # Verify with exact LP
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # If no exact solution found yet, get one from best smooth result
        if best_center is None:
            beta_r = best_smooth_beta
            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
            if S_exact < best_S:
                best_S = S_exact
                best_beta = beta_r
                best_center = c_exact
        
        # Final polish: use exact (non-smooth) objective with tight tolerances
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    max_proj = exact_max_proj(pts, beta, cx, cy)
                    return (max_proj + APOTHEM) / APOTHEM
                
                # More diverse starting points for polish with finer perturbations
                d = 0.005
                d2 = 0.01
                d3 = 0.02
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + d],
                    [best_center[0], best_center[1], best_beta - d],
                    [best_center[0], best_center[1], best_beta + d2],
                    [best_center[0], best_center[1], best_beta - d2],
                    [best_center[0], best_center[1], best_beta + d3],
                    [best_center[0], best_center[1], best_beta - d3],
                    [best_center[0] + d, best_center[1], best_beta],
                    [best_center[0] - d, best_center[1], best_beta],
                    [best_center[0], best_center[1] + d, best_beta],
                    [best_center[0], best_center[1] - d, best_beta],
                    [best_center[0] + d, best_center[1] + d, best_beta],
                    [best_center[0] - d, best_center[1] - d, best_beta],
                    [best_center[0] + d2, best_center[1] + d2, best_beta],
                    [best_center[0] - d2, best_center[1] - d2, best_beta],
                    [best_center[0] + d, best_center[1] - d, best_beta],
                    [best_center[0] - d, best_center[1] + d, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-16})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    # SLSQP polish from current best
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res2 = minimize(objective, [best_center[0], best_center[1], best_beta], 
                                       method='SLSQP', options={'maxiter': 10000, 'ftol': 1e-18})
                        if res2.fun < best_S:
                            best_S = res2.fun
                            best_center = (res2.x[0], res2.x[1])
                            best_beta = ((res2.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    except Exception:
                        pass
                    
                    # Additional Nelder-Mead polish from SLSQP result
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res3 = minimize(objective, [best_center[0], best_center[1], best_beta],
                                       method='Nelder-Mead',
                                       options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-16})
                        if res3.fun < best_S:
                            best_S = res3.fun
                            best_center = (res3.x[0], res3.x[1])
                            best_beta = ((res3.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    except Exception:
                        pass
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick estimate to prioritize promising ones
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        # Try multiple beta values for better estimate
        best_est = float('inf')
        for beta_try in [0.0, math.pi/12, math.pi/6, math.pi/24, 5*math.pi/24]:
            S_est, c_est, _ = calc_S_for_beta(pts, beta_try)
            if S_est < best_est:
                best_est = S_est
        return best_est

    estimated_scores = [(quick_estimate(ax), ax) for ax in candidates]
    estimated_scores.sort(key=lambda x: x[0])
    
    # Only fully optimize the top candidates within time budget
    # Each optimization takes ~2-5s, so we can do maybe 20-30 in 145s
    # Be less aggressive with skipping - rotation can improve things significantly
    for rank, (_, axial) in enumerate(estimated_scores):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Only skip if estimate is clearly much worse (rotation can help by ~5-10%)
        if rank > 12 and estimated_scores[rank][0] > best_global_S * 1.03:
            continue
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Final high-precision polish on the best configuration found
    if best_global_center is not None and best_global_pts is not None:
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def final_objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    max_proj = exact_max_proj(best_global_pts, beta, cx, cy)
                    return (max_proj + APOTHEM) / APOTHEM
                
                # Ultra-fine perturbations around the best solution
                fine_d = 0.001
                fine_starts = [
                    [best_global_center[0], best_global_center[1], best_global_beta],
                    [best_global_center[0], best_global_center[1], best_global_beta + fine_d],
                    [best_global_center[0], best_global_center[1], best_global_beta - fine_d],
                    [best_global_center[0] + fine_d, best_global_center[1], best_global_beta],
                    [best_global_center[0] - fine_d, best_global_center[1], best_global_beta],
                    [best_global_center[0], best_global_center[1] + fine_d, best_global_beta],
                    [best_global_center[0], best_global_center[1] - fine_d, best_global_beta],
                    [best_global_center[0] + fine_d, best_global_center[1] + fine_d, best_global_beta],
                    [best_global_center[0] - fine_d, best_global_center[1] - fine_d, best_global_beta],
                ]
                
                for fs in fine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res_f = minimize(final_objective, fs, method='Nelder-Mead',
                                   options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-18})
                    if res_f.fun < best_global_S:
                        best_global_S = res_f.fun
                        best_global_center = (res_f.x[0], res_f.x[1])
                        best_global_beta = ((res_f.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    # One more SLSQP pass with extremely tight tolerance
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res_fs = minimize(final_objective, [best_global_center[0], best_global_center[1], best_global_beta],
                                         method='SLSQP', options={'maxiter': 20000, 'ftol': 1e-20})
                        if res_fs.fun < best_global_S:
                            best_global_S = res_fs.fun
                            best_global_center = (res_fs.x[0], res_fs.x[1])
                            best_global_beta = ((res_fs.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    except Exception:
                        pass
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
    
    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-10
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
