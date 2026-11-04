import numpy as np
import time
import math
import random
from scipy.optimize import least_squares, minimize

N = 12
DEADLINE = 140.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Separation distance (positive) if disjoint, or negative penetration depth
    if overlapping. Uses the Separating Axis Theorem for two convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)

    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)

        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated

        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)

    return -min_overlap  # negative = overlapping

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of a regular hexagon centered at origin with the given
    rotation that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    """Find the outer-hexagon rotation that minimizes the enclosing side length."""
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

def support_function_value(cx, cy, rot_deg, normal_x, normal_y):
    """
    Support function of a unit regular hexagon centered at (cx,cy) with rotation rot_deg
    in the direction (normal_x, normal_y).
    For a regular hexagon with side length s, the support in direction n is:
    s * max over the 6 edge normals of (dot(n, e_k)) where e_k are the outward normals
    """
    # The 6 outward normals of a unit hexagon (side=1) at rotation 0
    base_angles = np.radians(np.array([90, 150, 210, 270, 330, 30]))
    rot_rad = np.radians(rot_deg)
    normals = base_angles + rot_rad
    nx = np.cos(normals)
    ny = np.sin(normals)
    
    # Support of unit hexagon in direction (normal_x, normal_y)
    # = max_k dot((normal_x, normal_y), (nx_k, ny_k)) * (side / cos(30deg)) ... 
    # Actually for a regular hexagon with inradius r = side * sqrt(3)/2,
    # support in direction n = r * max_k dot(n, e_k) where e_k are unit edge normals
    inradius = np.sqrt(3.0) / 2.0
    dots = normal_x * nx + normal_y * ny
    return cx * normal_x + cy * normal_y + inradius * np.max(dots)

def smooth_pairwise_penalty(params, beta=20.0):
    """
    Smooth penalty based on support function differences for non-overlap.
    Two convex bodies A and B are disjoint iff for all directions n,
    h_A(n) + h_B(-n) < 0 (in relative coordinates).
    We use the 12 edge normals of both hexagons as candidate separating directions.
    """
    xys_rots = params[:3 * N].reshape(N, 3)
    
    # For each pair, compute the minimum "separation" over the 12 relevant normals
    penetrations = []
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = xys_rots[i]
            xj, yj, rj = xys_rots[j]
            
            # Relative position
            dx = xi - xj
            dy = yi - yj
            
            # The 6 normals for hex i and 6 for hex j (12 total candidate axes)
            angles_i = np.radians(np.array([90, 150, 210, 270, 330, 30])) + np.radians(ri)
            angles_j = np.radians(np.array([90, 150, 210, 270, 330, 30])) + np.radians(rj)
            
            all_angles = np.concatenate([angles_i, angles_j])
            nx = np.cos(all_angles)
            ny = np.sin(all_angles)
            
            # Support of hex i in direction n
            dots_i = nx * np.cos(angles_i) + ny * np.sin(angles_i)
            h_i = np.sqrt(3.0)/2.0 * np.max(dots_i, axis=1)
            
            # Support of hex j in direction -n
            dots_j = -nx * np.cos(angles_j) + -ny * np.sin(angles_j)
            h_j = np.sqrt(3.0)/2.0 * np.max(dots_j, axis=1)
            
            # Separation in direction n: dx*nx + dy*ny - h_i - h_j
            # If this is positive for some n, they are separated
            sep = dx * nx + dy * ny - h_i - h_j
            min_sep = np.min(sep)
            
            if min_sep < 0:
                penetrations.append(-min_sep)
    
    if len(penetrations) == 0:
        return 0.0
    
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return float(np.sum(p_arr ** 2))
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return float(beta * lse)

def objective_combined(params, beta=50.0):
    """Outer side length + smooth overlap penalty."""
    xys_rots = params[:3 * N].reshape(N, 3)
    outer_rot = params[3 * N]

    penalty = smooth_pairwise_penalty(params, beta=beta)

    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    L = outer_hex_side(all_points, outer_rot)

    return L + 100.0 * penalty

def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    """Hexagonal lattice of unit hexagons (odd-row offset) with an optional gap."""
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
    """1 center + 6 ring + 5 outer, with an optional gap."""
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
    """12 hexagons in a ring around the center, each rotated to face outward."""
    pts = []
    R = 2.5 + gap
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        angle_deg = math.degrees(theta)
        pts.append([x, y, angle_deg])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def is_valid(params, tol=1e-6):
    """True if no pair of inner hexagons overlaps."""
    xys_rots = params[:3 * N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]

    for i in range(N):
        for j in range(i + 1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def get_L(params):
    """Outer side length for a configuration (assumed valid)."""
    xys_rots = params[:3 * N].reshape(N, 3)
    outer_rot = params[3 * N]
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    return outer_hex_side(all_points, outer_rot)

def rotate_points(pts, ang_deg):
    a = np.radians(ang_deg)
    c, s = np.cos(a), np.sin(a)
    out = pts.copy()
    out[:, 0] = pts[:, 0] * c - pts[:, 1] * s
    out[:, 1] = pts[:, 0] * s + pts[:, 1] * c
    return out

def golden_search(f, lo, hi, tol=1e-4, max_iter=60):
    """Minimize a scalar function f on [lo, hi] via golden-section search."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    m = 0.5 * (a + b)
    return f(m), m

def composite_body_search(best_params, time_budget=30.0):
    """
    Directed low-dimensional surrogate: treat the 12 hexagons as a single
    composite rigid body and minimize the bounding-hexagon side length over the
    body's global orientation (and an auxiliary uniform spacing scale) with a
    golden-section search. Returns (L, params) or (None, None).
    """
    start = time.time()

    def body_L(theta_deg, scale=1.0):
        if time.time() - start > time_budget:
            return float('inf')
        xys_rots = best_params[:3 * N].reshape(N, 3)
        new_pts = rotate_points(xys_rots[:, :2], theta_deg) * scale
        new_pts -= np.mean(new_pts, axis=0)
        new_xys = np.column_stack((new_pts, xys_rots[:, 2] + theta_deg))
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in new_xys])
        return outer_hex_side(all_points, 0.0)

    # Coarse 1D golden search over the body orientation.
    L1, theta1 = golden_search(lambda t: body_L(t, 1.0), 0.0, 60.0, tol=1e-3, max_iter=45)
    if L1 >= float('inf'):
        return None, None

    # Auxiliary uniform spacing scale around 1.0 (shrinking only, to stay valid).
    L2, scale = golden_search(lambda s: body_L(theta1, s), 0.90, 1.00, tol=1e-4, max_iter=30)
    if L2 >= float('inf'):
        L2, scale = L1, 1.0

    # Build the transformed configuration and re-optimize the outer rotation.
    xys_rots = best_params[:3 * N].reshape(N, 3)
    new_pts = rotate_points(xys_rots[:, :2], theta1) * scale
    new_pts -= np.mean(new_pts, axis=0)
    new_xys = np.column_stack((new_pts, xys_rots[:, 2] + theta1))
    new_params = np.concatenate((new_xys.flatten(), [0.0]))

    if not is_valid(new_params):
        return None, None

    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in new_xys])
    L_opt, rot_opt = optimize_outer_rotation(all_points, 0.0, n_samples=720)
    new_params[3 * N] = rot_opt

    return L_opt, new_params

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    best_L = float('inf')
    best_params = None

    # Diverse, valid initial configurations (small gap guarantees no overlap).
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))

    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)

    outer_rots = [0.0, 30.0, 15.0, 45.0]

    # Pair residuals for least_squares (penalize penetration only).
    def pair_residuals(params):
        xys_rots = params[:3 * N].reshape(N, 3)
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        res = []
        for i in range(N):
            for j in range(i + 1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                res.append(-d if d < 0 else 0.0)
        return np.array(res)

    # ---- Phase 1: seed + per-config gradient / local polish ----
    for cfg in configs:
        if time_left() < 25:
            break

        for orot in outer_rots:
            if time_left() < 25:
                break

            init_params = np.concatenate((cfg.flatten(), [orot]))

            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()

            if time_left() < 30:
                continue

            # least_squares: drive penetrations to zero (keeps config valid).
            try:
                res_ls = least_squares(
                    pair_residuals,
                    init_params,
                    method='trf',
                    max_nfev=150,
                    ftol=1e-8,
                    gtol=1e-8,
                )
                if is_valid(res_ls.x):
                    L_val = get_L(res_ls.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_ls.x.copy()
            except Exception:
                pass

            # Nelder-Mead polish on the smooth penalized objective.
            if time_left() < 40:
                continue
            try:
                start_p = best_params if best_params is not None else init_params
                res_nm = minimize(
                    lambda p: objective_combined(p, beta=50.0),
                    start_p,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True},
                )
                if is_valid(res_nm.x):
                    L_val = get_L(res_nm.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
            except Exception:
                pass

    # ---- Phase 2: directed composite-rigid-body golden-section search ----
    if best_params is not None and time_left() > 45:
        try:
            L_body, params_body = composite_body_search(best_params, time_budget=30.0)
            if L_body is not None and L_body < best_L:
                best_L = L_body
                best_params = params_body.copy()
        except Exception:
            pass

    # ---- Phase 3: outer-rotation optimization ----
    if best_params is not None and time_left() > 30:
        try:
            xys_rots = best_params[:3 * N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3 * N], n_samples=1440)
            if L_opt < best_L:
                best_L = L_opt
                best_params = best_params.copy()
                best_params[3 * N] = rot_opt
        except Exception:
            pass

    # ---- Phase 4: high-precision polish passes ----
    if best_params is not None and time_left() > 20:
        try:
            res_final = minimize(
                lambda p: objective_combined(p, beta=30.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
            )
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 12:
        try:
            res_lbfgs = minimize(
                lambda p: objective_combined(p, beta=20.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000},
            )
            if is_valid(res_lbfgs.x):
                L_val = get_L(res_lbfgs.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_lbfgs.x.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 8:
        try:
            res_polish = minimize(
                lambda p: objective_combined(p, beta=100.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
            )
            if is_valid(res_polish.x):
                L_val = get_L(res_polish.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_polish.x.copy()
        except Exception:
            pass

    # ---- Phase 5: final outer-rotation sweep + micro-polish ----
    if best_params is not None and time_left() > 5:
        try:
            xys_rots = best_params[:3 * N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3 * N], n_samples=2880)
            if L_opt < best_L:
                best_L = L_opt
                best_params = best_params.copy()
                best_params[3 * N] = rot_opt
        except Exception:
            pass

    if best_params is not None and time_left() > 2:
        try:
            xys_rots = best_params[:3 * N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            cur_rot = best_params[3 * N]
            fine_rots = np.linspace(cur_rot - 0.1, cur_rot + 0.1, 400)
            best_local_L = best_L
            best_local_rot = cur_rot
            for rot in fine_rots:
                L = outer_hex_side(all_points, rot)
                if L < best_local_L:
                    best_local_L = L
                    best_local_rot = rot
            if best_local_L < best_L:
                best_L = best_local_L
                best_params = best_params.copy()
                best_params[3 * N] = best_local_rot
        except Exception:
            pass

    # Fallback (should not be reached for valid seeds).
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), float(L_val), 0.0

    xys_rots = best_params[:3 * N].reshape(N, 3)
    outer_rot = best_params[3 * N]

    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)

    return inner, center, side, angle
