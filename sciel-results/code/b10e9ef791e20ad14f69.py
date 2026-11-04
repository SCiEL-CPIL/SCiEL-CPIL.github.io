import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
SQRT3 = np.sqrt(3.0)

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns separation distance if not overlapping (positive),
    or negative penetration depth if overlapping.
    """
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)

    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)

        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap

        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)

    return -min_overlap

def outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / SQRT3) * max_proj

def smooth_overlap_penalty(params, beta=40.0):
    xys_rots = params[:3 * N].reshape(N, 3)
    centers = xys_rots[:, :2]

    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        vertices_list.append(get_hex_vertices(cx, cy, 1.0, rot))

    penetrations = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)

    if len(penetrations) == 0:
        return 0.0

    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return float(np.sum(p_arr ** 2))
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def is_valid(params, tol=1e-7):
    xys_rots = params[:3 * N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    for i in range(N):
        for j in range(i + 1, N):
            if sat_overlap(vertices_list[i], vertices_list[j]) < -tol:
                return False
    return True

def get_L(params):
    xys_rots = params[:3 * N].reshape(N, 3)
    outer_rot = params[3 * N]
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    return outer_hex_side(all_points, outer_rot)

def make_hex_grid(rows, cols, gap=0.0, rot=0.0):
    pts = []
    spacing_x = SQRT3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    if len(pts) == 0:
        pts = np.array([[0, 0, rot]])
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring(gap=0.0, rot=0.0):
    pts = []
    d1 = SQRT3 + gap
    d2 = 2 * SQRT3 + gap
    pts.append([0, 0, rot])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), rot])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), rot])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def objective(params):
        if time.time() - start_time > DEADLINE - 3:
            return 1e6
        penalty = smooth_overlap_penalty(params, beta=40.0)
        xys_rots = params[:3 * N].reshape(N, 3)
        outer_rot = params[3 * N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        return L + 50.0 * penalty

    best_L = float('inf')
    best_params = None

    # Diverse initial configurations
    gap = 0.02
    configs = []
    for rows, cols in [(4, 3), (3, 4), (2, 6), (6, 2), (3, 3), (4, 4)]:
        configs.append(make_hex_grid(rows, cols, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring(gap=gap, rot=30.0))
    configs.append(make_hex_grid(4, 3, gap=gap, rot=30.0))
    configs.append(make_hex_grid(3, 4, gap=gap, rot=30.0))

    # Add a few more with larger gaps for safety
    configs.append(make_hex_grid(4, 3, gap=0.1))
    configs.append(make_ring(gap=0.1))

    outer_rots = [0.0, 30.0, 15.0, 45.0, 7.5, 22.5]

    for cfg in configs:
        if time_left() < 15:
            break
        for orot in outer_rots:
            if time_left() < 10:
                break

            init_params = np.concatenate((cfg.flatten(), [orot]))

            # Check initial validity
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()

            # Phase 1: L-BFGS-B
            if time_left() > 12:
                try:
                    res = minimize(objective, init_params, method='L-BFGS-B',
                                   options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 400})
                    if is_valid(res.x):
                        L_val = get_L(res.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res.x.copy()
                except Exception:
                    pass

            # Phase 2: Nelder-Mead polish
            if time_left() > 15:
                try:
                    polish_start = best_params if best_params is not None else init_params
                    res_nm = minimize(objective, polish_start, method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                    if is_valid(res_nm.x):
                        L_val = get_L(res_nm.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res_nm.x.copy()
                except Exception:
                    pass

    # Final high-precision polish
    if best_params is not None and time_left() > 10:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        inner = [[float(4.0 * k), 0.0, 0.0] for k in range(N)]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3 * N].reshape(N, 3)
    outer_rot = best_params[3 * N]
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    return inner, center, side, angle
