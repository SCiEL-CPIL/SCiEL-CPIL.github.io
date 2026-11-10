# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

# Precomputed unit hexagon (angle 0)
_UNIT_COS = np.cos(np.deg2rad(60 * np.arange(6)))
_UNIT_SIN = np.sin(np.deg2rad(60 * np.arange(6)))
_UNIT_VERTS = np.column_stack([_UNIT_COS, _UNIT_SIN])  # (6,2)
_UNIT_EDGE_NORMALS = np.array([
    [1.0, 0.0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2],
    [-1.0, 0.0], [-0.5, -np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2],
])  # (6,2) outward edge normals for angle-0 unit hexagon

def _rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])

def _rot_mat(angle_deg):
    return _rot2(np.deg2rad(angle_deg))

def _min_side_and_grad(params, outer_angle_deg):
    """
    params: (38,) = [12 centers (24), 12 angles (12), outer center (2)]
    Returns (s, grad_params).
    s = min side of regular outer hexagon (angle outer_angle_deg, centered at oc)
    containing all rotated unit inner hexagons.
    """
    inner_pos = params[:24].reshape(12, 2)
    inner_ang = params[24:36]
    oc = params[36:38]

    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)

    R = _rot_mat(inner_ang)  # (12,2,2)
    verts = inner_pos[:, None, :] + (R @ _UNIT_VERTS.T).T  # (12,6,2)
    verts_flat = verts.reshape(72, 2)
    rel = verts_flat - oc[None, :]
    projs = rel @ normals.T  # (72,6)
    max_projs = np.max(projs, axis=0)  # (6,)
    max_inradius = float(np.max(max_projs))
    if max_inradius < 0:
        max_inradius = 0.0
    s = 2.0 * max_inradius / np.sqrt(3.0)

    grad = np.zeros_like(params)
    if max_inradius > 1e-12:
        argmax_per_normal = np.argmax(projs, axis=0)  # (6,)
        active_normal = int(np.argmax(max_projs))
        active_vertex_idx = int(argmax_per_normal[active_normal])
        nvec = normals[active_normal]
        hex_idx = active_vertex_idx // 6
        vert_idx = active_vertex_idx % 6
        coef = 2.0 / np.sqrt(3.0)
        grad[2 * hex_idx:2 * hex_idx + 2] = coef * nvec
        grad[36:38] = -coef * nvec
        u = _UNIT_VERTS[vert_idx]
        perp_u = np.array([-u[1], u[0]])
        dv_da = R[hex_idx] @ perp_u
        grad[24 + hex_idx] = coef * float(dv_da @ nvec)
    return s, grad

def _overlap_penalty(params, outer_angle_deg):
    """Sum of squared penetration distances between all inner hexagon pairs."""
    inner_pos = params[:24].reshape(12, 2)
    inner_ang = params[24:36]
    R = _rot_mat(inner_ang)  # (12,2,2)
    norms = _UNIT_EDGE_NORMALS[None, :, :] @ R  # (12,6,2) rotated normals
    pen = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(6):
                n = norms[i, k]
                c = inner_pos[j] - inner_pos[i]
                d = np.dot(c, n)
                if d < 0:
                    pen += d * d
            for k in range(6):
                n = norms[j, k]
                c = inner_pos[i] - inner_pos[j]
                d = np.dot(c, n)
                if d < 0:
                    pen += d * d
    return pen

def _obj(params, outer_angle_deg, lam):
    s, _ = _min_side_and_grad(params, outer_angle_deg)
    return s + lam * _overlap_penalty(params, outer_angle_deg)

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
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
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
        inner_verts_list.append(verts)
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _make_hex_lattice(n_rings, spacing):
    positions = [[0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        for i in range(6 * ring):
            angle = np.deg2rad(60 * (i // ring + 1) + (i % ring) * (60.0 / ring))
            positions.append([ring * spacing * np.cos(angle),
                              ring * spacing * np.sin(angle)])
    positions = np.array(positions[:12])
    positions -= positions.mean(axis=0)
    return positions

def _make_grid(rows, cols, dx, dy):
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2.0) * dx
            y = (r - (rows - 1) / 2.0) * dy
            if r % 2 == 1:
                x += dx / 2
            positions.append([x, y])
    positions = np.array(positions[:12])
    positions -= positions.mean(axis=0)
    return positions

def _params_from_pos(pos):
    p = np.zeros(38)
    p[:24] = pos.flatten()
    return p

def _anneal(params, outer_angle_deg, lam, rng, deadline):
    """Simulated annealing with gradient steps to escape overlaps/local minima."""
    params = params.copy()
    best = params.copy()
    best_s = _min_side_and_grad(params, outer_angle_deg)[0]
    T = 0.05
    for _ in range(4000):
        if time.time() > deadline:
            break
        s, grad = _min_side_and_grad(params, outer_angle_deg)
        noise = rng.normal(0, 0.15, size=38)
        cand = params - 0.05 * grad + noise
        s2, _ = _min_side_and_grad(cand, outer_angle_deg)
        if s2 < s or rng.random() < np.exp(-(s2 - s) / max(T, 1e-9)):
            params = cand
        if s2 < best_s:
            best_s = s2
            best = cand.copy()
        T *= 0.995
    return best, best_s

def _nelder_mead(params, outer_angle_deg, lam, maxiter, deadline):
    def obj(p):
        if time.time() > deadline:
            return 1e10
        return _obj(p, outer_angle_deg, lam)
    try:
        res = minimize(obj, params, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        return res.x, res.fun
    except Exception:
        return params, _obj(params, outer_angle_deg, lam)

def hexagon_packing_12():
    rng = np.random.default_rng(20240607)
    start = time.time()
    deadline = start + 585.0  # safety margin before 630s

    configs = []
    for sp in [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
        configs.append(_make_hex_lattice(2, sp))
    sqrt3 = np.sqrt(3)
    for dx, dy in [(2.0, sqrt3 * 0.8), (2.2, sqrt3 * 0.9), (2.4, sqrt3),
                   (2.6, sqrt3 * 1.1), (2.8, sqrt3 * 1.2), (3.0, sqrt3 * 1.3)]:
        configs.append(_make_grid(3, 4, dx, dy))
        configs.append(_make_grid(4, 3, dx, dy))
        configs.append(_make_grid(2, 6, dx, dy))
        configs.append(_make_grid(6, 2, dx, dy))
    for _ in range(12):
        base = _make_hex_lattice(2, 2.5).copy()
        base += rng.normal(0, 0.6, size=(12, 2))
        configs.append(base)

    outer_angles = [0, 30, 60, 90, 120, 150]
    lam = 50.0

    best_s = np.inf
    best_params = None
    best_angle = 0

    # Phase 1: anneal + Nelder-Mead from many starts
    for angle_deg in outer_angles:
        if time.time() > deadline - 90:
            break
        for pos in configs:
            if time.time() > deadline - 90:
                break
            p0 = _params_from_pos(pos)
            p_a, _ = _anneal(p0, angle_deg, lam, rng, deadline - 90)
            p_nm, _ = _nelder_mead(p_a, angle_deg, lam, 4000, deadline - 90)
            s_nm = _min_side_and_grad(p_nm, angle_deg)[0]
            if s_nm < best_s:
                best_s = s_nm
                best_params = p_nm
                best_angle = angle_deg

    # Phase 2: fine polish of the best
    if best_params is not None and time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            p_nm, _ = _nelder_mead(best_params, best_angle, lam, 20000, deadline - 20)
            s_nm = _min_side_and_grad(p_nm, best_angle)[0]
            if s_nm < best_s:
                best_s = s_nm
                best_params = p_nm

    def build(p, angle_deg):
        inner_pos = p[:24].reshape(12, 2)
        inner_ang = p[24:36]
        oc = p[36:38]
        inner = np.zeros((12, 3))
        inner[:, 0] = inner_pos[:, 0]
        inner[:, 1] = inner_pos[:, 1]
        inner[:, 2] = inner_ang
        outer = np.array([oc[0], oc[1], float(angle_deg)])
        s = _min_side_and_grad(p, angle_deg)[0]
        return inner, outer, s

    if best_params is not None:
        inner, outer, s_final = build(best_params, best_angle)
    else:
        inner, outer, s_final = None, None, np.inf

    if inner is None or not _validate(inner, outer, s_final):
        pos = _make_hex_lattice(2, 2.5)
        inner = np.zeros((12, 3))
        inner[:, 0] = pos[:, 0]
        inner[:, 1] = pos[:, 1]
        outer = np.array([0.0, 0.0, 0.0])
        pfb = _params_from_pos(pos)
        s_final = _min_side_and_grad(pfb, 0.0)[0]
        for grow in [0.0, 0.1, 0.3, 0.6, 1.0, 1.5]:
            if _validate(inner, outer, s_final + grow):
                s_final = s_final + grow
                break

    return inner, outer, s_final

# EVOLVE-BLOCK-END
