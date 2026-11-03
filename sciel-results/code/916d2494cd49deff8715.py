import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_normals(beta):
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        centered = pts - np.array(center)
        projs = centered @ normals.T
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta, center=None):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')
        normals = get_normals(beta)
        n_pts = len(pts)
        A_ub = np.zeros((12 * n_pts, 3))
        b_ub = np.zeros(12 * n_pts)
        for k in range(6):
            nx, ny = normals[k]
            idx1 = 2 * k * n_pts
            idx2 = idx1 + n_pts
            A_ub[idx1:idx1+n_pts, 0] = -nx
            A_ub[idx1:idx1+n_pts, 1] = -ny
            A_ub[idx1:idx1+n_pts, 2] = -1.0
            b_ub[idx1:idx1+n_pts] = -(pts[:, 0] * nx + pts[:, 1] * ny)
            A_ub[idx2:idx2+n_pts, 0] = nx
            A_ub[idx2:idx2+n_pts, 1] = ny
            A_ub[idx2:idx2+n_pts, 2] = -1.0
            b_ub[idx2:idx2+n_pts] = pts[:, 0] * nx + pts[:, 1] * ny
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def hex_dist(a, b):
        """Minimum distance between two unit hexagon centers for non-overlap."""
        da = a - b
        da = ((da % (math.pi / 3.0)) + math.pi / 3.0) % (math.pi / 3.0)
        # For two unit hexagons, min center distance = 2 * apothem = sqrt(3)
        # This is exact for aligned hexagons; for rotated ones it's a lower bound
        return 2.0 * APOTHEM

    def non_overlap_penalty(pts, temp=0.05):
        """Smooth penalty for overlaps. Returns 0 if no overlaps."""
        n = len(pts)
        min_dist = 2.0 * APOTHEM  # sqrt(3)
        violations = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    violations.append(min_dist - d)
        if not violations:
            return 0.0
        v = np.array(violations)
        max_v = float(np.max(v))
        shifted = (v - max_v) / temp
        shifted = np.clip(shifted, -100, 100)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return temp * (log_sum + max_v / temp)

    def smooth_objective(params, temp=0.05):
        """Combined objective: outer side length + overlap penalty."""
        # params: [x0, y0, x1, y1, ..., x10, y10, beta]
        pts = params[:2*N].reshape(N, 2)
        beta = params[2*N]
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        
        normals = get_normals(beta)
        center = np.mean(pts, axis=0)
        D = compute_D(pts, center, normals)
        S = (D + APOTHEM) / APOTHEM
        
        penalty = non_overlap_penalty(pts, temp)
        
        return S + 10.0 * penalty

    def exact_objective(params):
        """Exact objective: outer side length with overlap check."""
        pts = params[:2*N].reshape(N, 2)
        beta = params[2*N]
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        
        # Check overlaps
        min_dist = 2.0 * APOTHEM
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist - 1e-7:
                    return S + 100.0 * (min_dist - d)
        
        S, _, _ = calc_S_for_beta(pts, beta)
        return S

    # Initial configurations
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    configs = []
    
    # Hexagonal cluster
    c1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    configs.append(axial_to_cartesian(c1))
    
    # 3x3 + 2
    c2 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    configs.append(axial_to_cartesian(c2))
    
    # 4+3+3
    c3 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)]
    configs.append(axial_to_cartesian(c3))
    
    # 3x3 + 2 top
    c4 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    configs.append(axial_to_cartesian(c4))
    
    # 2+4+4+1
    c5 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)]
    configs.append(axial_to_cartesian(c5))
    
    # 3+4+3+1
    c6 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)]
    configs.append(axial_to_cartesian(c6))
    
    # 4+4+3
    c7 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    configs.append(axial_to_cartesian(c7))
    
    # 5+6
    c8 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    configs.append(axial_to_cartesian(c8))
    
    # 3+2+3+3
    c9 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    configs.append(axial_to_cartesian(c9))
    
    # 2+3+2+3+1
    c10 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)]
    configs.append(axial_to_cartesian(c10))
    
    # 2+3+3+2+1
    c11 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(1,4)]
    configs.append(axial_to_cartesian(c11))
    
    # 3+3+3+1+1
    c12 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,3)]
    configs.append(axial_to_cartesian(c12))

    best_S = float('inf')
    best_params = None

    # Multi-start optimization
    for cfg_idx, cfg in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Center the config
        cfg_centered = cfg - np.mean(cfg, axis=0)
        
        # Try multiple betas
        betas = [0.0, math.pi/12, math.pi/6, math.pi/24, math.pi/8]
        
        for beta0 in betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Initial params
            init = np.zeros(2*N + 1)
            init[:2*N] = cfg_centered.ravel()
            init[2*N] = beta0
            
            # Smooth optimization with annealing
            temps = [0.3, 0.1, 0.05, 0.02, 0.01]
            current_params = init.copy()
            
            for temp in temps:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                def obj(p, t=temp):
                    return smooth_objective(p, t)
                
                try:
                    res = minimize(obj, current_params, method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res.fun < 1e10:
                        current_params = res.x
                except Exception:
                    pass
            
            # Final exact evaluation
            try:
                pts = current_params[:2*N].reshape(N, 2)
                beta = ((current_params[2*N] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                # Check overlaps
                min_dist = 2.0 * APOTHEM
                overlap = False
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_dist - 1e-6:
                            overlap = True
                            break
                    if overlap:
                        break
                
                if not overlap:
                    S, center, D = calc_S_for_beta(pts, beta)
                    if S < best_S:
                        best_S = S
                        best_params = current_params.copy()
            except Exception:
                pass

    # Polish with more precise optimization
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Try a few more refinement steps
        for _ in range(3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, 0.005), best_params,
                             method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < 1e10:
                    pts = res.x[:2*N].reshape(N, 2)
                    beta = ((res.x[2*N] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    min_dist = 2.0 * APOTHEM
                    ok = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            if np.linalg.norm(pts[i] - pts[j]) < min_dist - 1e-6:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        S, _, _ = calc_S_for_beta(pts, beta)
                        if S < best_S:
                            best_S = S
                            best_params = res.x
            except Exception:
                break

    # Fallback
    if best_params is None:
        cfg = configs[0] - np.mean(configs[0], axis=0)
        best_params = np.zeros(2*N + 1)
        best_params[:2*N] = cfg.ravel()
        best_params[2*N] = 0.0
        pts = cfg
        S, _, _ = calc_S_for_beta(pts, 0.0)
        best_S = S

    # Extract final result
    pts = best_params[:2*N].reshape(N, 2)
    beta = ((best_params[2*N] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
    angle_deg = math.degrees(beta)

    # Compute exact outer hexagon
    S, center, D = calc_S_for_beta(pts, beta)
    if center is None:
        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        normals = get_normals(beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = (pts[:, 0] - center[0]) * nx + (pts[:, 1] - center[1]) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        S = (max_proj + APOTHEM) / APOTHEM

    # Small safety scale
    SCALE = 1.0 + 1e-10
    cx, cy = center
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    # Recompute S with scaled points
    normals = get_normals(beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (np.array(pts_scaled)[:, 0] - cx) * nx + (np.array(pts_scaled)[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    S_final = (max_proj + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    return inner, [float(cx), float(cy)], float(S_final), float(angle_deg)
