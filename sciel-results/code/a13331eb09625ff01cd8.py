import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3.0)
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagon_vertices(cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        verts = []
        for k in range(6):
            angle = rot + k * math.pi / 3.0
            verts.append((cx + side * math.cos(angle), cy + side * math.sin(angle)))
        return verts

    # Precompute candidate axial configurations (rigid hexagonal lattice layouts)
    candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
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
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(0,1),(0,-1),(1,1)],
        [(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(1,1),(-1,1),(0,1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)],
        [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)],
    ]

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation."""
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
                cx, cy = res.x[0], res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
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

    def smooth_overlap_penalty(centers, rotations, temperature=0.02, min_dist=2.0):
        """Log-sum-exp soft penalty for hexagon pair overlaps."""
        n = len(centers)
        penalties = []
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                violation = min_dist - dist
                if violation > 0:
                    penalties.append(violation)
        if not penalties:
            return 0.0
        pen_arr = np.array(penalties)
        max_val = np.max(pen_arr)
        if max_val > 50:
            return 1e10
        shifted = pen_arr / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return temperature * log_sum

    def optimize_orientation_only(pts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        n_grid = 61
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
                                     options={'maxiter': 100, 'xatol': 1e-12})
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
                                     options={'maxiter': 200, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        return best_S, best_beta, best_center

    def optimize_shape(pts):
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
                temperatures = [0.03, 0.01]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)
                    start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-11})
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
                delta_vals = [0.0, 0.001, -0.001, 0.003, -0.003, 0.005, -0.005]
                polish_starts = []
                for db in delta_vals:
                    for dcx in [0.0, 0.001, -0.001]:
                        for dcy in [0.0, 0.001, -0.001]:
                            polish_starts.append([best_center[0] + dcx, best_center[1] + dcy, best_beta + db])
                seen = set()
                unique_starts = []
                for p in polish_starts:
                    key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
                    if key not in seen:
                        seen.add(key)
                        unique_starts.append(p)
                for init in unique_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass
        return best_S, best_beta, best_center

    # === LATTICE RELAXATION: deformable mesh with smooth overlap penalty ===
    def lattice_relaxation(init_pts, init_rotations, beta_init=0.0, cx_init=None, cy_init=None):
        """Optimize positions, rotations, center, and container rotation jointly."""
        n = len(init_pts)
        if cx_init is None:
            cx_init = float(np.mean(init_pts[:, 0]))
        if cy_init is None:
            cy_init = float(np.mean(init_pts[:, 1]))

        # Parameters: [x1,y1,rot1, ..., xn,yn,rotn, cx, cy, beta]
        def unpack(params):
            centers = np.array([[params[3*i], params[3*i+1]] for i in range(n)])
            rots = np.array([params[3*i+2] for i in range(n)])
            cx = params[3*n]
            cy = params[3*n+1]
            beta = params[3*n+2]
            return centers, rots, cx, cy, beta

        def smooth_objective(params):
            centers, rots, cx, cy, beta = unpack(params)
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            # Container size (smooth)
            S_val = smooth_calc_S(centers, cx, cy, beta, temperature=0.03)
            # Overlap penalty
            overlap = smooth_overlap_penalty(centers, rots, temperature=0.02, min_dist=2.0)
            # Return combined: minimize S + heavy penalty for overlap
            return S_val + 50.0 * overlap

        # Build initial parameter vector
        init_params = []
        for i in range(n):
            init_params.extend([init_pts[i, 0], init_pts[i, 1], init_rotations[i]])
        init_params.extend([cx_init, cy_init, beta_init])

        best_S = float('inf')
        best_params = None

        # Multi-start Nelder-Mead with annealing
        n_starts = 5
        for s in range(n_starts):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Perturb initial parameters for diversity
            if s == 0:
                start_params = list(init_params)
            else:
                start_params = []
                for i in range(n):
                    jitter_x = (np.random.random() - 0.5) * 0.1 * s
                    jitter_y = (np.random.random() - 0.5) * 0.1 * s
                    start_params.extend([init_pts[i, 0] + jitter_x, init_pts[i, 1] + jitter_y, init_rotations[i]])
                start_params.extend([cx_init, cy_init, beta_init + (s-1)*0.01])

            try:
                res = minimize(smooth_objective, start_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
                if res.fun < 1e8:
                    centers, rots, cx, cy, beta = unpack(res.x)
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    # Check if overlaps are resolved
                    overlap_check = smooth_overlap_penalty(centers, rots, temperature=0.01, min_dist=2.0)
                    if overlap_check < 0.01:
                        S_exact = exact_S(centers, cx, cy, beta)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = list(res.x)
            except Exception:
                pass

        # Final polish with exact objective
        if best_params is not None:
            try:
                def exact_objective(params):
                    centers, rots, cx, cy, beta = unpack(params)
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    # Hard constraint: if any overlap, return large penalty
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = centers[i, 0] - centers[j, 0]
                            dy = centers[i, 1] - centers[j, 1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 2.0 - 1e-9:
                                return 1e6 + (2.0 - dist)
                    return exact_S(centers, cx, cy, beta)

                res = minimize(exact_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13,
                                      'adaptive': True})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = list(res.x)
            except Exception:
                pass

        if best_params is None:
            return float('inf'), None, None, None, None

        centers, rots, cx, cy, beta = unpack(best_params)
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        return best_S, centers, rots, (cx, cy), beta

    # Evaluate all candidates: first with rigid optimization, then with relaxation
    best_global_S = float('inf')
    best_global_pts = None
    best_global_rots = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    scored_candidates = []
    for axial in candidates:
        est = quick_estimate(axial)
        scored_candidates.append((est, axial))
    scored_candidates.sort(key=lambda x: x[0])

    # Phase 1: Rigid optimization (fast)
    for idx, (est, axial) in enumerate(scored_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_rots = np.zeros(N)
            best_global_beta = beta
            best_global_center = center

    # Phase 2: Lattice relaxation on top candidates (slower but can find non-rigid optima)
    n_relax = min(8, len(scored_candidates))
    for idx in range(n_relax):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        est, axial = scored_candidates[idx]
        pts = axial_to_cartesian(axial)
        rots = np.zeros(N)
        S_relax, centers_relax, rots_relax, center_relax, beta_relax = lattice_relaxation(
            pts, rots, beta_init=best_global_beta,
            cx_init=best_global_center[0] if best_global_center else None,
            cy_init=best_global_center[1] if best_global_center else None
        )
        if S_relax < best_global_S and centers_relax is not None:
            # Verify no overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers_relax[i, 0] - centers_relax[j, 0]
                    dy = centers_relax[i, 1] - centers_relax[j, 1]
                    if math.sqrt(dx*dx + dy*dy) < 2.0 - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_global_S = S_relax
                best_global_pts = centers_relax
                best_global_rots = rots_relax
                best_global_beta = beta_relax
                best_global_center = center_relax

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_rots = np.zeros(N)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale points slightly outward from center to ensure containment
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
    for i in range(N):
        x, y = pts_scaled[i]
        rot_deg = math.degrees(best_global_rots[i]) if best_global_rots is not None else float(angle_deg)
        inner.append([float(x), float(y), float(rot_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
