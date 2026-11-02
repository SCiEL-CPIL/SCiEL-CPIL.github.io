import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

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

    def generate_hexes(params):
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

    def objective(params):
        if time.time() > deadline:
            return 1e10
        hexes = generate_hexes(params)
        s_min = compute_min_side(hexes)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 20.0
        return s_min + penalty

    def smooth_objective(params):
        if time.time() > deadline:
            return 1e10
        hexes = generate_hexes(params)
        s_min = compute_min_side(hexes)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += np.log(1.0 + np.exp(15.0 * overlap))
        return s_min + penalty

    bounds = [
        (-5.0, 5.0),
        (-5.0, 5.0),
        (0.0, np.pi/3),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (0.0, np.pi/3),
    ]

    best_s = np.inf
    best_params = None

    # Phase 1: Global search with differential evolution
    try:
        res = differential_evolution(
            smooth_objective,
            bounds,
            maxiter=200,
            popsize=25,
            tol=1e-7,
            seed=42,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Phase 2: Local refinement with multiple starts
    if best_params is not None:
        try:
            res_local = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 2000}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Phase 3: Random restarts with local optimization
    if time.time() < deadline - 15:
        np.random.seed(123)
        n_restarts = 10
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            if best_params is not None and r % 2 == 0:
                p = best_params + np.random.normal(0, 0.3, 6)
            else:
                spacing = 2.0 * np.sqrt(3.0)
                r1 = np.random.uniform(0.5, spacing)
                a1 = np.random.uniform(0, 2*np.pi)
                r2 = np.random.uniform(spacing*0.5, spacing*1.5)
                a2 = np.random.uniform(0, 2*np.pi)
                rot1 = np.random.uniform(0, np.pi/3)
                rot2 = np.random.uniform(0, np.pi/3)
                p = np.array([r1*np.cos(a1), r1*np.sin(a1), rot1,
                              r2*np.cos(a2), r2*np.sin(a2), rot2])
            p = np.clip(p, [-5, -5, 0, -5, -5, 0], [5, 5, np.pi/3, 5, 5, np.pi/3])
            try:
                res_local = minimize(
                    objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 1500}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if best_params is not None and time.time() < deadline - 3:
        try:
            res_polish = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxfun': 5000}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = np.array([1.5, 0.0, 0.0, 2.5, 0.0, 0.0])

    hexes = generate_hexes(best_params)
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
