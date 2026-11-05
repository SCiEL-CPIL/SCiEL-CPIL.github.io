import numpy as np
import time
import math
import random
from scipy.optimize import least_squares, minimize

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

# Precompute rotation matrices for efficiency
def get_rot_matrix(rot_deg):
    rad = np.radians(rot_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s], [s, c]])

# Unit hexagon vertices at origin with 0 rotation
_UNIT_HEX_VERTS = get_hex_vertices(0, 0, 1.0, 0.0)

def get_hex_vertices_fast(cx, cy, rot_deg):
    """Fast hexagon vertex computation using precomputed base and rotation matrix."""
    R = get_rot_matrix(rot_deg)
    return _UNIT_HEX_VERTS @ R.T + np.array([cx, cy])

def outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    best_L = float('inf')
    best_rot = init_rot
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    local_rots = np.linspace(best_rot - 2.0, best_rot + 2.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    return best_L, best_rot

def sat_overlap(v1, v2):
    """Separating axis theorem for two convex polygons.
    Returns positive gap if separated, negative penetration depth if overlapping."""
    n1, n2 = len(v1), len(v2)
    axes = []
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1) % n1]
        e = p2 - p1
        norm = np.linalg.norm(e)
        if norm > 1e-12:
            n = np.array([-e[1], e[0]]) / norm
            axes.append(n)
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i+1) % n2]
        e = p2 - p1
        norm = np.linalg.norm(e)
        if norm > 1e-12:
            n = np.array([-e[1], e[0]]) / norm
            axes.append(n)
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2:
            return min2 - max1
        if max2 < min1:
            return min1 - max2
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return -min_overlap

def smooth_overlap_penalty(params, beta=30.0):
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    vertices_list = [get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots]
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    if not penetrations:
        return 0.0
    p_arr = np.array(penetrations)
    max_pen = np.max(p_arr)
    if max_pen < 1e-12:
        return 0.0
    # Use log-sum-exp for numerical stability
    shifted = beta * (p_arr - max_pen)
    shifted = np.clip(shifted, -50, 50)
    log_sum = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return log_sum

def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring12(gap=0.0):
    pts = []
    R = 2.2 + gap
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        angle_deg = math.degrees(theta)
        pts.append([x, y, angle_deg])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_compact_cluster(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    for c in range(3):
        pts.append([c * dx, 0, 0.0])
    for c in range(4):
        pts.append([c * dx + dx/2, dy, 0.0])
    for c in range(3):
        pts.append([c * dx, 2*dy, 0.0])
    for c in range(2):
        pts.append([c * dx + dx/2, 3*dy, 0.0])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_diamond(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    for c in range(2):
        pts.append([c * dx + dx/2, 0, 0.0])
    for c in range(3):
        pts.append([c * dx, dy, 0.0])
    for c in range(4):
        pts.append([c * dx + dx/2, 2*dy, 0.0])
    for c in range(3):
        pts.append([c * dx, 3*dy, 0.0])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def is_valid(params, tol=1e-6):
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots]
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def get_L(params):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots])
    return outer_hex_side(all_points, outer_rot)

def objective_combined(params, penalty_weight=80.0, beta=30.0):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    vertices_list = [get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots]
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(xys_rots[i, :2] - xys_rots[j, :2])
            if d < 2.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    if penetrations:
        p_arr = np.array(penetrations)
        max_pen = np.max(p_arr)
        if max_pen < 1e-12:
            penalty = 0.0
        else:
            shifted = beta * (p_arr - max_pen)
            shifted = np.clip(shifted, -50, 50)
            penalty = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    else:
        penalty = 0.0
    all_points = np.vstack(vertices_list)
    L = outer_hex_side(all_points, outer_rot)
    return L + penalty_weight * penalty

def residuals_least_squares(params, beta=30.0):
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    vertices_list = [get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots]
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    if not penetrations:
        return np.zeros(1)
    p_arr = np.array(penetrations)
    max_pen = np.max(p_arr)
    if max_pen < 1e-12:
        return np.zeros(1)
    shifted = beta * (p_arr - max_pen)
    shifted = np.clip(shifted, -50, 50)
    log_sum = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return np.array([log_sum])

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    best_L = float('inf')
    best_params = None

    configs = []
    for gap in [0.05, 0.02]:
        configs.append(make_hex_grid(4, 3, gap=gap))
        configs.append(make_hex_grid(3, 4, gap=gap))
        configs.append(make_hex_grid(2, 6, gap=gap))
        configs.append(make_hex_grid(6, 2, gap=gap))
        configs.append(make_ring(gap=gap))
        configs.append(make_ring12(gap=gap))
        configs.append(make_compact_cluster(gap=gap))
        configs.append(make_diamond(gap=gap))

    for base_cfg in configs[:8]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
        rotated2 = base_cfg.copy()
        rotated2[:, 2] = 15.0
        configs.append(rotated2)

    outer_rots = [0.0, 30.0, 15.0, 45.0]

    for cfg in configs:
        if time_left() < 20:
            break
        for orot in outer_rots:
            if time_left() < 20:
                break
            init_params = np.concatenate((cfg.flatten(), [orot]))
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            if time_left() < 25:
                continue
            try:
                res_ls = least_squares(
                    residuals_least_squares,
                    init_params,
                    method='trf',
                    max_nfev=300,
                    ftol=1e-10,
                    gtol=1e-10
                )
                if is_valid(res_ls.x):
                    L_val = get_L(res_ls.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_ls.x.copy()
            except Exception:
                pass

    # Honeycomb rigid-body optimization: treat all hexagons as a single rigid body
    # and optimize their positions relative to each other while keeping them in a honeycomb pattern
    if best_params is not None and time_left() > 45:
        # Try different honeycomb configurations with small gaps
        honeycomb_configs = []
        for gap in [0.0, 0.01, 0.02, 0.03]:
            for rows, cols in [(3, 4), (4, 3), (2, 6), (6, 2)]:
                cfg = make_hex_grid(rows, cols, gap=gap)
                honeycomb_configs.append(cfg)
        
        for hcfg in honeycomb_configs:
            if time_left() < 35:
                break
            for orot in [0.0, 30.0]:
                if time_left() < 35:
                    break
                init_params = np.concatenate((hcfg.flatten(), [orot]))
                if not is_valid(init_params):
                    continue
                try:
                    # Use Nelder-Mead to optimize the entire configuration
                    res_nm = minimize(
                        objective_combined,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if is_valid(res_nm.x):
                        L_val = get_L(res_nm.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res_nm.x.copy()
                except Exception:
                    pass

    # Simulated annealing phase
    if best_params is not None and time_left() > 40:
        current_params = best_params.copy()
        current_L = best_L
        T_start = 0.1
        T_end = 0.001
        n_sa_iter = 800
        sa_start_time = time.time()
        sa_budget = min(50, time_left() - 30)

        for it in range(n_sa_iter):
            if time.time() - sa_start_time > sa_budget:
                break
            T = T_start * (T_end / T_start) ** (it / max(1, n_sa_iter - 1))

            # Perturb
            new_params = current_params.copy()
            # Move 1-3 random hexagons with adaptive step size
            n_move = random.choice([1, 2, 3])
            step_scale = 1.0 + 0.5 * (1 - it / n_sa_iter)
            for _ in range(n_move):
                idx = random.randint(0, N - 1)
                new_params[3*idx] += random.gauss(0, 0.3 * step_scale)
                new_params[3*idx + 1] += random.gauss(0, 0.3 * step_scale)
                new_params[3*idx + 2] += random.gauss(0, 5.0 * step_scale)
            # Slightly perturb outer rotation
            new_params[3*N] += random.gauss(0, 2.0)

            if not is_valid(new_params):
                continue

            new_L = get_L(new_params)
            delta = new_L - current_L

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_params = new_params
                current_L = new_L
                if new_L < best_L:
                    best_L = new_L
                    best_params = new_params.copy()

    # Nelder-Mead polish after SA
    if best_params is not None and time_left() > 35:
        try:
            res_nm = minimize(
                objective_combined,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if is_valid(res_nm.x):
                L_val = get_L(res_nm.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
        except Exception:
            pass

    # Optimize outer rotation
    if best_params is not None and time_left() > 25:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=1440)
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass

    # Final high-precision polish
    if best_params is not None and time_left() > 20:
        try:
            res_final = minimize(
                objective_combined,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
        except Exception:
            pass

    # Final outer rotation optimization
    if best_params is not None and time_left() > 8:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    return inner, center, side, angle
