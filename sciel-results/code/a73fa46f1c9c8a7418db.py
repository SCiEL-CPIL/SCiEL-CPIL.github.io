import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=1440):
    best_L = float('inf')
    best_rot = init_rot
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    local_rots = np.linspace(best_rot - 3.0, best_rot + 3.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    return best_L, best_rot

def sat_overlap(v1, v2):
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
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
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    return -min_overlap

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

def make_compact_cluster(gap=0.0):
    """Create a compact 3x4 grid with slight rotation for better packing"""
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(3):
        for c in range(4):
            x = c * spacing_x
            y = r * spacing_y
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_star_pattern(gap=0.0):
    """Create a star-like pattern with central hex and surrounding ones"""
    pts = []
    sqrt3 = np.sqrt(3)
    d_inner = sqrt3 + gap
    d_outer = 2.5 * sqrt3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d_inner * np.cos(ang), d_inner * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d_outer * np.cos(ang), d_outer * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def perturb_config(config, magnitude=0.1):
    """Apply small random perturbations to a configuration"""
    perturbed = config.copy()
    for i in range(N):
        perturbed[i, 0] += random.uniform(-magnitude, magnitude)
        perturbed[i, 1] += random.uniform(-magnitude, magnitude)
        perturbed[i, 2] += random.uniform(-5, 5)
    return perturbed

def is_valid(params, tol=1e-7):
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def smooth_max(values, beta=50.0):
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m < -50.0 / beta:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def objective_combined(params, beta=50.0):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < 0:
                penalty += smooth_max(np.array([0.0, -d]), beta=beta) ** 2
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    L = (2.0 / np.sqrt(3.0)) * smooth_max(max_projs, beta=beta)
    return L + 50.0 * penalty

def objective_precise(params, beta=100.0):
    """High-precision objective for final polish"""
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < 0:
                penalty += smooth_max(np.array([0.0, -d]), beta=beta) ** 2
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    L = (2.0 / np.sqrt(3.0)) * smooth_max(max_projs, beta=beta)
    return L + 100.0 * penalty

def construct():
    start_time = time.time()
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    configs = []
    for gap in [0.0, 0.02, 0.05, 0.1]:
        configs.append(make_hex_grid(4, 3, gap=gap))
        configs.append(make_hex_grid(3, 4, gap=gap))
        configs.append(make_hex_grid(2, 6, gap=gap))
        configs.append(make_hex_grid(6, 2, gap=gap))
        configs.append(make_ring(gap=gap))
        configs.append(make_compact_cluster(gap=gap))
        configs.append(make_star_pattern(gap=gap))
    for base_cfg in configs[:8]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    # Add some perturbed versions for diversity
    for base_cfg in configs[:4]:
        for _ in range(2):
            configs.append(perturb_config(base_cfg, magnitude=0.05))
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 10.0, 20.0]
    
    for cfg in configs:
        if time_left() < 15:
            break
        for orot in outer_rots:
            if time_left() < 15:
                break
            init_params = np.concatenate((cfg.flatten(), [orot]))
            if is_valid(init_params):
                xys_rots = init_params[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            if time_left() < 25:
                continue
            try:
                res_nm = minimize(
                    lambda p, b=30.0: objective_combined(p, b),
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                if is_valid(res_nm.x):
                    xys_rots = res_nm.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_nm.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
            except Exception:
                pass
    
    if best_params is not None and time_left() > 25:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    if best_params is not None and time_left() > 20:
        try:
            res_final = minimize(
                lambda p: objective_precise(p),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if is_valid(res_final.x):
                xys_rots = res_final.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, res_final.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time_left() > 10:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=5760)
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
