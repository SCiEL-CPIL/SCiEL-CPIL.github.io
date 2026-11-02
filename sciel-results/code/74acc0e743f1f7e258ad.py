import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
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

    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3)])
    # extra diverse shapes
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(0,-2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(2,-1)])

    def center_and_D(pts, beta):
        """Fast exact center (midpoint of extreme projections) and D. No LP."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = pts @ normals.T
        max_p = projs.max(axis=0)
        min_p = projs.min(axis=0)
        D = np.max((max_p - min_p) * 0.5)
        c = np.zeros(2)
        for k in range(6):
            c += (max_p[k] + min_p[k]) * 0.5 * normals[k]
        return c, D

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pc = pts - np.array([cx, cy])
        projs = pc @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pc = pts - np.array([cx, cy])
        projs = pc @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def collision_penalty(pts, min_dist=2.0 * APOTHEM):
        n = len(pts)
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    pen += (min_dist - d) ** 2
        return pen

    def check_overlap(pts, min_dist=2.0 * APOTHEM):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts[i] - pts[j]) < min_dist - 1e-6:
                    return False
        return True

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, pts

        # Stage 1: grid over beta with fast center formula
        n_grid = 49
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            c, D = center_and_D(pts, beta)
            S = (D + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = (float(c[0]), float(c[1]))

        if best_center is None:
            return float('inf'), 0.0, None, pts

        # Stage 2: refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def fbeta(b):
                    c, D = center_and_D(pts, b)
                    return (D + APOTHEM) / APOTHEM
                res = minimize_scalar(fbeta, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-11})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    c, _ = center_and_D(pts, best_beta)
                    best_center = (float(c[0]), float(c[1]))
        except Exception:
            pass

        # Stage 3: joint position + center + rotation polish with smooth barrier
        def joint_obj(params, temperature=0.02):
            n_pts = len(pts)
            flat = params
            positions = flat[:2 * n_pts].reshape(n_pts, 2)
            cx = flat[2 * n_pts]
            cy = flat[2 * n_pts + 1]
            beta = flat[2 * n_pts + 2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            S = smooth_calc_S(positions, cx, cy, beta, temperature=temperature)
            min_dist = 2.0 * APOTHEM
            pen = 0.0
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_dist:
                        pen += 100.0 * (min_dist - d) ** 2
            return S + pen

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.05, 0.02, 0.01]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    start_betas = [best_beta, best_beta + 0.04, best_beta - 0.04, 0.0, math.pi / 12]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = pts.flatten().tolist() + [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(joint_obj, init, args=(temp,), method='Nelder-Mead',
                                         options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
                            if res.fun < 1e10:
                                n_pts = len(pts)
                                opt_pts = res.x[:2 * n_pts].reshape(n_pts, 2)
                                cx_r, cy_r, beta_r = res.x[2 * n_pts], res.x[2 * n_pts + 1], res.x[2 * n_pts + 2]
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                if check_overlap(opt_pts):
                                    S_exact = exact_S(opt_pts, cx_r, cy_r, beta_r)
                                    if S_exact < best_S:
                                        best_S = S_exact
                                        best_beta = beta_r
                                        best_center = (float(cx_r), float(cy_r))
                                        best_pts = opt_pts
                        except Exception:
                            pass
        except Exception:
            pass

        # Stage 4: high-precision exact polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(best_pts, cx, cy, beta)
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                    [best_center[0], best_center[1], best_beta - 0.01],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0] - 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0], best_center[1] - 0.01, best_beta],
                    [best_center[0] + 0.005, best_center[1] + 0.005, best_beta],
                    [best_center[0] - 0.005, best_center[1] - 0.005, best_beta],
                ]
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 4000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center, best_pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center, opt_pts = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = opt_pts if opt_pts is not None else pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        c, D = center_and_D(best_global_pts, 0.0)
        best_global_center = (float(c[0]), float(c[1]))
        best_global_beta = 0.0
        best_global_S = (D + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

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
