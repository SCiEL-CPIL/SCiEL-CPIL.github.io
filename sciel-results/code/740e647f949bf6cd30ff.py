import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        return np.min(overlaps)

    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    outer_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    def compute_min_side(hexes):
        max_proj = 0.0
        for h in hexes:
            proj = h @ outer_normals.T
            max_proj = max(max_proj, np.max(proj))
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_smooth_min_side(hexes, beta=50.0):
        all_projs = []
        for h in hexes:
            proj = h @ outer_normals.T
            all_projs.append(proj)
        all_projs = np.concatenate(all_projs, axis=1)
        scaled = beta * all_projs
        max_scaled = np.max(scaled, axis=0)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=0)) / beta) / beta
        return np.max(soft_max) * 2.0 / np.sqrt(3.0)

    def compute_overlap_penalty(hexes, k=50.0):
        penalty = 0.0
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += np.log1p(np.exp(k * overlap))
        return penalty

    def objective_exact(params, gen_func, bounds):
        hexes = gen_func(params)
        s_min = compute_min_side(hexes)
        penalty = compute_overlap_penalty(hexes, k=50.0)
        return s_min + penalty

    def objective_smooth(params, gen_func, bounds, beta=50.0, k=50.0):
        hexes = gen_func(params)
        s_min = compute_smooth_min_side(hexes, beta=beta)
        penalty = compute_overlap_penalty(hexes, k=k)
        return s_min + penalty

    # 6-fold symmetry: 2 orbits of 6
    def generate_hexes_6fold(params):
        x1, y1, a1, x2, y2, a2 = params
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s = np.sin(angle)
            x1r = x1 * c - y1 * s
            y1r = x1 * s + y1 * c
            a1r = a1 + angle
            x2r = x2 * c - y2 * s
            y2r = x2 * s + y2 * c
            a2r = a2 + angle
            hexes.append(hex_vertices(x1r, y1r, 1.0, a1r))
            hexes.append(hex_vertices(x2r, y2r, 1.0, a2r))
        return hexes

    bounds_6fold = [
        (-5.0, 5.0), (-5.0, 5.0), (0.0, np.pi/3),
        (-5.0, 5.0), (-5.0, 5.0), (0.0, np.pi/3)
    ]

    # 3-fold symmetry: 4 orbits of 3
    def generate_hexes_3fold(params):
        hexes = []
        for i in range(4):
            xi, yi, ai = params[3*i], params[3*i+1], params[3*i+2]
            for k in range(3):
                angle = np.radians(120 * k)
                c = np.cos(angle)
                s = np.sin(angle)
                xr = xi * c - yi * s
                yr = xi * s + yi * c
                ar = ai + angle
                hexes.append(hex_vertices(xr, yr, 1.0, ar))
        return hexes

    bounds_3fold = []
    for _ in range(4):
        bounds_3fold.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, np.pi/3)])

    best_s = np.inf
    best_params_6 = None
    best_params_3 = None
    best_gen_func = generate_hexes_6fold

    # Phase 1: 6-fold symmetry optimization
    if time.time() < deadline - 30:
        try:
            res = differential_evolution(
                lambda p: objective_smooth(p, generate_hexes_6fold, bounds_6fold, beta=30.0, k=20.0),
                bounds_6fold,
                maxiter=200,
                popsize=40,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params_6 = res.x.copy()
                best_gen_func = generate_hexes_6fold
                
            # Local refinement
            if best_params_6 is not None:
                res_local = minimize(
                    lambda p: objective_exact(p, generate_hexes_6fold, bounds_6fold),
                    best_params_6,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params_6 = res_local.x.copy()
                    best_gen_func = generate_hexes_6fold
        except Exception:
            pass

    # Phase 2: 3-fold symmetry optimization
    if time.time() < deadline - 20:
        try:
            res = differential_evolution(
                lambda p: objective_smooth(p, generate_hexes_3fold, bounds_3fold, beta=30.0, k=20.0),
                bounds_3fold,
                maxiter=100,
                popsize=30,
                tol=1e-8,
                seed=123,
                polish=False
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params_3 = res.x.copy()
                best_gen_func = generate_hexes_3fold
                
            if best_params_3 is not None:
                res_local = minimize(
                    lambda p: objective_exact(p, generate_hexes_3fold, bounds_3fold),
                    best_params_3,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params_3 = res_local.x.copy()
                    best_gen_func = generate_hexes_3fold
        except Exception:
            pass

    # Phase 3: Final polish with smooth objective
    if best_gen_func == generate_hexes_6fold and best_params_6 is not None and time.time() < deadline - 5:
        try:
            res_smooth = minimize(
                lambda p: objective_smooth(p, generate_hexes_6fold, bounds_6fold, beta=100.0, k=50.0),
                best_params_6,
                method='L-BFGS-B',
                bounds=bounds_6fold,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_smooth.fun < best_s:
                best_s = res_smooth.fun
                best_params_6 = res_smooth.x.copy()
        except Exception:
            pass
    elif best_gen_func == generate_hexes_3fold and best_params_3 is not None and time.time() < deadline - 5:
        try:
            res_smooth = minimize(
                lambda p: objective_smooth(p, generate_hexes_3fold, bounds_3fold, beta=100.0, k=50.0),
                best_params_3,
                method='L-BFGS-B',
                bounds=bounds_3fold,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_smooth.fun < best_s:
                best_s = res_smooth.fun
                best_params_3 = res_smooth.x.copy()
        except Exception:
            pass

    # Select best params
    if best_gen_func == generate_hexes_6fold:
        final_params = best_params_6
    else:
        final_params = best_params_3

    if final_params is None:
        final_params = np.array([1.5, 0.0, 0.0, 2.5, 0.0, 0.0])
        best_gen_func = generate_hexes_6fold

    hexes = best_gen_func(final_params)
    inner = []
    for h in hexes:
        cx = np.mean(h[:, 0])
        cy = np.mean(h[:, 1])
        edge = h[1] - h[0]
        angle = np.arctan2(edge[1], edge[0])
        inner.append([cx, cy, np.degrees(angle) % 360])

    s_final = compute_min_side(hexes)
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
