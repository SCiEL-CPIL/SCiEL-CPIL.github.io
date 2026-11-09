import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600.0
    safety_margin = 15.0

    # --- Validation and scoring helpers ---
    def is_valid(cx, cy, s):
        if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy)) and np.all(np.isfinite(s))):
            return False
        if not np.all(s > 0):
            return False
        # Boundary check
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1.0 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1.0 + tol:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    def score_solution(cx, cy, s):
        return float(np.sum(s))

    # --- Baseline: 3x3 grid + 1 ---
    # This gives sum = 10/3 ≈ 3.333
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_base = (cols + 0.5) * side
    cy_base = (rows + 0.5) * side
    s_base = np.full(n, side, dtype=np.float64)
    
    best_score = score_solution(cx_base, cy_base, s_base)
    best_cx = cx_base.copy()
    best_cy = cy_base.copy()
    best_s = s_base.copy()

    # --- Objective function for optimization ---
    # We minimize -sum(s) + penalty
    def objective(params):
        cx = params[:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        
        # Penalty for boundary violations
        pen = 0.0
        for i in range(n):
            pen += max(0, -tol - (cx[i] - s[i]/2))**2
            pen += max(0, (cx[i] + s[i]/2) - (1.0 + tol))**2
            pen += max(0, -tol - (cy[i] - s[i]/2))**2
            pen += max(0, (cy[i] + s[i]/2) - (1.0 + tol))**2
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                # If overlapping, penalize the amount of overlap squared
                ox = max(0, t - dx)
                oy = max(0, t - dy)
                if ox > 0 and oy > 0:
                    pen += (ox * oy)**2

        return -np.sum(s) + 100.0 * pen

    # --- Bounds ---
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.01, 1.0)] * n

    # --- Initial Configurations ---
    # 1. Grid
    init_configs = [np.hstack([cx_base, cy_base, s_base])]

    # 2. 4 large corners, 6 small middle
    # Large side ~0.35, Small side ~0.25
    s_l = 0.35
    s_m = 0.25
    cfg2 = np.zeros(3*n)
    # Corners
    cfg2[0] = s_l/2; cfg2[n] = s_l/2; cfg2[2*n] = s_l
    cfg2[1] = 1-s_l/2; cfg2[n+1] = s_l/2; cfg2[2*n+1] = s_l
    cfg2[2] = s_l/2; cfg2[n+2] = 1-s_l/2; cfg2[2*n+2] = s_l
    cfg2[3] = 1-s_l/2; cfg2[n+3] = 1-s_l/2; cfg2[2*n+3] = s_l
    # Middle 6: arranged in a 2x3 grid in the center?
    # Center area is roughly [0.35, 0.65] x [0.35, 0.65] -> size 0.3x0.3? No, too small.
    # Let's just scatter them in the center and let optimizer fix it.
    # Actually, a better initial guess for the middle 6 is to place them in a strip or grid.
    # Let's try a 2x3 grid in the center square [0.25, 0.75]x[0.25, 0.75]
    # Side would be 0.5/3 = 0.166? Or 0.5/2 = 0.25?
    # Let's place 6 squares of side 0.25 in a 2x3 grid in the center.
    # x positions: 0.25, 0.5, 0.75 (centers) -> width 0.75? No.
    # If side is 0.25, total width for 3 is 0.75. Centers at 0.125, 0.375, 0.625?
    # Let's just use random-ish valid positions for the middle 6.
    positions = [
        (0.5, 0.3), (0.5, 0.5), (0.5, 0.7),
        (0.3, 0.5), (0.7, 0.5), (0.5, 0.5) # one overlaps, optimizer will fix
    ]
    for i in range(6):
        idx = 4 + i
        cfg2[idx] = positions[i][0]
        cfg2[n+idx] = positions[i][1]
        cfg2[2*n+idx] = s_m
    init_configs.append(cfg2)

    # 3. 2 large top/bottom, 8 small sides?
    # Or just run DE from multiple random starts too.

    # --- Optimization Phase 1: Differential Evolution ---
    # Use DE to find a good global layout
    if time.time() < deadline - 120:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-6,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_score * -1 + 1e-6: # Rough check, better to validate
                cx, cy, s = de_result.x[:n], de_result.x[n:2*n], de_result.x[2*n:3*n]
                if is_valid(cx, cy, s):
                    sc = score_solution(cx, cy, s)
                    if sc > best_score:
                        best_score = sc
                        best_cx = cx.copy()
                        best_cy = cy.copy()
                        best_s = s.copy()
        except Exception:
            pass

    # --- Optimization Phase 2: Local Refinement (Nelder-Mead) ---
    # Refine the best solution found so far, and also try refining the initial configs
    # Nelder-Mead is good for smooth local improvements.
    
    def nm_objective(params):
        # Use a stricter penalty to ensure validity during refinement
        cx = params[:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        
        pen = 0.0
        # Hard constraint violations get high penalty
        for i in range(n):
            if cx[i] - s[i]/2 < -tol: pen += 1000 * (-tol - (cx[i] - s[i]/2))**2
            if cx[i] + s[i]/2 > 1.0 + tol: pen += 1000 * ((cx[i] + s[i]/2) - (1.0 + tol))**2
            if cy[i] - s[i]/2 < -tol: pen += 1000 * (-tol - (cy[i] - s[i]/2))**2
            if cy[i] + s[i]/2 > 1.0 + tol: pen += 1000 * ((cy[i] + s[i]/2) - (1.0 + tol))**2
        
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0 - tol
                if dx < t and dy < t:
                    # Overlap violation
                    # Push apart
                    ox = t - dx
                    oy = t - dy
                    pen += 1000 * (ox**2 + oy**2)
        
        return -np.sum(s) + pen

    start_points = [np.hstack([best_cx, best_cy, best_s])]
    # Add initial configs as start points for NM
    for cfg in init_configs:
        start_points.append(cfg)
    
    # Add some random perturbations of the best solution
    np.random.seed(123)
    for _ in range(5):
        pert = np.random.normal(0, 0.05, 3*n)
        sp = np.hstack([best_cx, best_cy, best_s]) + pert
        # Clip to bounds
        sp[:n] = np.clip(sp[:n], 0, 1)
        sp[n:2*n] = np.clip(sp[n:2*n], 0, 1)
        sp[2*n:3*n] = np.clip(sp[2*n:3*n], 0.01, 1)
        start_points.append(sp)

    for sp in start_points:
        if time.time() > deadline - safety_margin:
            break
        try:
            nm_result = minimize(
                nm_objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            cx, cy, s = nm_result.x[:n], nm_result.x[n:2*n], nm_result.x[2*n:3*n]
            if is_valid(cx, cy, s):
                sc = score_solution(cx, cy, s)
                if sc > best_score:
                    best_score = sc
                    best_cx = cx.copy()
                    best_cy = cy.copy()
                    best_s = s.copy()
        except Exception:
            pass

    # --- Final Check and Return ---
    # Ensure we return the best valid solution found
    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)
    
    return centers, sides
