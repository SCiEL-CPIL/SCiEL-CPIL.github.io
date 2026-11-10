# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of L-shaped corridor placements to maximize the area
    of the intersection (the sofa that can be moved through the corridor).
    """
    start_time = time.time()
    deadline = start_time + 580

    # Sofa grid points
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = sofa_points.shape[0]

    def is_inside_base(u, v):
        cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return cond1 | cond2

    def generate_placements(path):
        placements = [(0.0, 0.0, 0.0)]
        for i in range(len(path) - 1):
            x1, y1, a1 = path[i]
            x2, y2, a2 = path[i + 1]
            m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
            for k in range(1, m + 1):
                t = k / m
                placements.append((x1 + t*(x2-x1), y1 + t*(y2-y1), a1 + t*(a2-a1)))
        if len(path) > 0:
            placements.append((path[-1][0], path[-1][1], path[-1][2]))
        placements.append((-2.0, 0.0, 90.0))
        return placements

    def count_inside_fast(points, placements):
        mask = np.ones(len(points), dtype=bool)
        for (cx, cy, a_deg) in placements:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            mask &= is_inside_base(u, v)
            if not np.any(mask):
                return 0
        return np.sum(mask)

    def circular_arc_path(n_ctrl, cx, cy, r, a_start, a_end):
        angles = np.linspace(a_start, a_end, n_ctrl)
        x = cx + r * np.cos(np.radians(angles) + np.pi/2)
        y = cy + r * np.sin(np.radians(angles) + np.pi/2)
        return np.column_stack([x, y, angles])

    def objective(params):
        if time.time() > deadline:
            return 1e10
        n_ctrl = len(params) // 3
        path = params.reshape(n_ctrl, 3).copy()
        path[:, 2] = np.clip(path[:, 2], 5, 85)
        for i in range(1, n_ctrl):
            if path[i, 2] <= path[i-1, 2] + 0.5:
                path[i, 2] = path[i-1, 2] + 0.5
        placements = generate_placements(path)
        count = count_inside_fast(sofa_points, placements)
        return -count

    n_ctrl = 8
    # Start with a good circular arc
    best_params = circular_arc_path(n_ctrl, -0.3, 0.3, 0.8, 10.0, 80.0).ravel().copy()
    bounds = [(-1.5, 0.5), (-0.5, 1.0), (5, 85)] * n_ctrl

    best_count = 0

    # Phase 1: Nelder-Mead
    try:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if res.fun < 1e9:
            best_count = -res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Phase 2: L-BFGS-B
    if time.time() < deadline - 120:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            if res2.fun < best_count - 0.5:
                best_count = -res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Try different arc centers/radii
    if time.time() < deadline - 60:
        for cx_t in [-0.2, -0.3, -0.4, -0.5]:
            for cy_t in [0.2, 0.3, 0.4]:
                for r_t in [0.6, 0.8, 1.0]:
                    if time.time() > deadline - 20:
                        break
                    arc_p = circular_arc_path(n_ctrl, cx_t, cy_t, r_t, 10.0, 80.0).ravel()
                    try:
                        res_a = minimize(objective, arc_p, method='Nelder-Mead',
                                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                        if res_a.fun < best_count - 0.5:
                            best_count = -res_a.fun
                            best_params = res_a.x.copy()
                    except Exception:
                        pass

    # Phase 4: Final polish
    if time.time() < deadline - 10:
        try:
            res_f = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res_f.fun < best_count - 0.5:
                best_count = -res_f.fun
                best_params = res_f.x.copy()
        except Exception:
            pass

    best_path = best_params.reshape(n_ctrl, 3).copy()
    for i in range(1, n_ctrl):
        if best_path[i, 2] <= best_path[i-1, 2]:
            best_path[i, 2] = best_path[i-1, 2] + 0.1
    best_path[:, 2] = np.clip(best_path[:, 2], 5, 85)

    if time.time() < deadline - 5:
        placements = generate_placements(best_path)
        final_count = count_inside_fast(sofa_points, placements)
        baseline_path = np.array([
            [-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0], [-0.4, 0.3, 50.0], [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0], [-0.7, 0.3, 86.0],
        ])
        bp = generate_placements(baseline_path)
        bc = count_inside_fast(sofa_points, bp)
        if final_count < bc:
            best_path = baseline_path

    best_path = np.nan_to_num(best_path, nan=0.0, posinf=0.0, neginf=0.0)
    return best_path

# EVOLVE-BLOCK-END
