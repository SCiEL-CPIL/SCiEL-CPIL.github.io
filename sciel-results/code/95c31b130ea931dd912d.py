import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_BUDGET = 120.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # ---- Build candidate axial coordinate sets ----
    base_sets = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4),(1,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(1,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(3,0)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,2),(1,3),(2,3),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(1,-1)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(1,4),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(2,2),(3,2)],
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def rotate_axial(axial_list, steps):
        result = []
        for q, r in axial_list:
            for _ in range(steps % 6):
                q, r = -r, q + r
            result.append((q, r))
        return result

    all_candidates = []
    seen = set()
    for base in base_sets:
        for rot in range(6):
            rotated = rotate_axial(base, rot)
            key = tuple(sorted(rotated))
            if key not in seen:
                seen.add(key)
                all_candidates.append(rotated)

    def fast_exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        max_proj = np.max(np.abs(projs))
        return (max_proj + APOTHEM) / APOTHEM

    def calc_S_for_beta(pts, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        n_pts = len(pts)

        A_ub_list = []
        b_ub_list = []
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            A_ub_list.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(projs)
            A_ub_list.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(-projs)

        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx, cy = res.x[0], res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    def optimize_shape(pts, current_best_S=float('inf')):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Phase 1: Coarse grid over beta
        for i in range(37):
            if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / 36
            S, center = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_S >= current_best_S * 1.001 and current_best_S < float('inf'):
            return best_S, best_beta, best_center

        # Phase 2: Bounded scalar refinement of beta
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Phase 3: Multi-start Nelder-Mead on exact objective
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return fast_exact_S(pts, cx, cy, beta)

                bcx, bcy = best_center
                bb = best_beta
                starts = [
                    [bcx, bcy, bb],
                    [bcx, bcy, bb + 0.02],
                    [bcx, bcy, bb - 0.02],
                    [bcx, bcy, bb + 0.05],
                    [bcx, bcy, bb - 0.05],
                    [bcx + 0.02, bcy, bb],
                    [bcx - 0.02, bcy, bb],
                    [bcx, bcy + 0.02, bb],
                    [bcx, bcy - 0.02, bb],
                    [bcx + 0.01, bcy + 0.01, bb],
                    [bcx - 0.01, bcy - 0.01, bb],
                    [bcx + 0.01, bcy - 0.01, bb],
                    [bcx - 0.01, bcy + 0.01, bb],
                    [bcx, bcy, 0.0],
                    [bcx, bcy, math.pi / 12],
                    [bcx, bcy, math.pi / 6],
                ]

                for init in starts:
                    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(objective, init, method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Phase 4: Final high-precision polish
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                def objective2(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return fast_exact_S(pts, cx, cy, beta)

                bcx, bcy = best_center
                bb = best_beta
                polish_starts = [
                    [bcx, bcy, bb],
                    [bcx + 0.005, bcy, bb],
                    [bcx - 0.005, bcy, bb],
                    [bcx, bcy + 0.005, bb],
                    [bcx, bcy - 0.005, bb],
                    [bcx, bcy, bb + 0.01],
                    [bcx, bcy, bb - 0.01],
                ]
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(objective2, init, method='Nelder-Mead',
                                     options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # ---- Main loop over all candidates ----
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(all_candidates):
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts, current_best_S=best_global_S)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(base_sets[0])
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

    # Tiny safety scale to avoid numerical overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    # Recompute exact S with safety scale
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
