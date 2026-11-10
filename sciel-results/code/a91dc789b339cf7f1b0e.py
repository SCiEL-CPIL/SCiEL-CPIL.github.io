# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute unit hexagon vertices (side=1, angle=0)
_ANGLES = np.deg2rad(60.0 * np.arange(6))
_UNIT_VERTS = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])  # (6,2)

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + _ANGLES
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_min_side(inner_positions, outer_center, outer_angle_deg):
    """Min side s of outer hex (center, angle) containing all inner unit hexagons."""
    outer_angle = np.deg2rad(outer_angle_deg)
    # outward normals of outer hex edges: angle = outer_angle + 30 + 60*i
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    # all inner vertices relative to outer center
    rel = inner_positions[:, None, :] + _UNIT_VERTS[None, :, :] - outer_center  # (12,6,2)
    proj = np.einsum('ijk,jk->ij', rel, normals)  # (12,6)
    max_inradius = np.max(proj)
    if max_inradius < 0:
        max_inradius = 0.0
    return 2.0 * max_inradius / np.sqrt(3.0)

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    if s < np.sqrt(12) - 1e-8:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            if abs(np.linalg.norm(verts[(k + 1) % 6] - verts[k]) - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _build_result(params, outer_angle_deg):
    inner_positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = inner_positions[:, 0]
    inner_hex_data[:, 1] = inner_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle_deg])
    s = _compute_min_side(inner_positions, outer_center, outer_angle_deg)
    return inner_hex_data, outer_hex_data, s

def _objective(params, outer_angle_deg):
    inner_positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    return _compute_min_side(inner_positions, outer_center, outer_angle_deg)

def _optimize_once(params0, outer_angle_deg, deadline):
    """Run gradient descent then Nelder-Mead refinement, return best params."""
    def obj(p):
        return _objective(p, outer_angle_deg)
    best = params0.copy()
    best_f = obj(best)
    # Gradient descent with adaptive step
    p = params0.copy()
    f = best_f
    lr = 0.05
    for it in range(3000):
        if time.time() > deadline:
            break
        g = np.zeros_like(p)
        eps = 1e-6
        for k in range(p.shape[0]):
            if time.time() > deadline:
                break
            pk = p.copy()
            pk[k] += eps
            g[k] = (obj(pk) - f) / eps
        gn = np.linalg.norm(g)
        if gn > 1e-14:
            step = lr / gn
        else:
            break
        p_new = p - np.clip(step, 0.0, 1.0) * g
        f_new = obj(p_new)
        if f_new < f:
            f = f_new
            p = p_new
            if f < best_f:
                best_f = f
                best = p.copy()
            lr = min(lr * 1.05, 0.3)
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    # Nelder-Mead refinement
    if time.time() < deadline:
        try:
            res = minimize(obj, best, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_f:
                best_f = res.fun
                best = res.x
        except Exception:
            pass
    return best, best_f

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0 - 40.0  # safety margin

    sqrt3 = np.sqrt(3)
    # Diverse initial configurations
    configs = []
    # 1) honeycomb 4x3
    dx = 2.5
    dy = sqrt3
    pos = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * dx
            y = (row - 1.0) * dy
            if row % 2 == 1:
                x += dx / 2
            pos.append([x, y])
    pos = np.array(pos)
    pos -= pos.mean(axis=0)
    configs.append(pos)
    # 2) ring of 6 + inner 6 (hexagonal lattice)
    pos2 = []
    r = 2.0
    for i in range(6):
        a = np.deg2rad(60 * i)
        pos2.append([r * np.cos(a), r * np.sin(a)])
    for i in range(6):
        a = np.deg2rad(30 + 60 * i)
        pos2.append([r * 0.5 * np.cos(a), r * 0.5 * np.sin(a)])
    pos2 = np.array(pos2)
    pos2 -= pos2.mean(axis=0)
    configs.append(pos2)
    # 3) 3x4 grid
    pos3 = []
    for row in range(4):
        for col in range(3):
            x = (col - 1.0) * dx
            y = (row - 1.5) * dy
            if row % 2 == 1:
                x += dx / 2
            pos3.append([x, y])
    pos3 = np.array(pos3)
    pos3 -= pos3.mean(axis=0)
    configs.append(pos3)

    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    best_params = None
    best_s = np.inf
    best_angle = 0.0

    # Optimize each (config, angle) combo, prioritizing by time
    combos = []
    for cfg in configs:
        for ang in angles:
            combos.append((cfg, ang))
    # interleave for diversity: do each config with a few angles first
    order = []
    for ci in range(len(configs)):
        for ai in range(len(angles)):
            order.append((ci, ai))

    for (ci, ai) in order:
        if time.time() > deadline:
            break
        cfg = configs[ci]
        ang = angles[ai]
        params0 = np.concatenate([cfg.flatten(), [0.0, 0.0]])
        p, f = _optimize_once(params0, ang, deadline)
        if f < best_s:
            best_s = f
            best_params = p.copy()
            best_angle = ang

    inner_hex_data, outer_hex_data, s_final = _build_result(best_params, best_angle)

    if not _validate(inner_hex_data, outer_hex_data, s_final):
        # Fallback to a safe honeycomb
        pos_fb = configs[0].copy()
        inner_hex_data = np.zeros((12, 3))
        inner_hex_data[:, 0] = pos_fb[:, 0]
        inner_hex_data[:, 1] = pos_fb[:, 1]
        inner_hex_data[:, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(pos_fb, np.array([0.0, 0.0]), 0.0) + 0.05

    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
