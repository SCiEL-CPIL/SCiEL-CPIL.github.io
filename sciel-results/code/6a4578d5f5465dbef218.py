# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(60 * np.arange(6))
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    return False


def _hexagons_separated(inner_data):
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        n = _hex_normals(a)
        verts.append(v)
        norms.append(n)
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _contains_in_hex(inner_data, outer_data, side):
    ox, oy, oa = outer_data
    outer_v = _hex_vertices(ox, oy, side, oa)
    tol = -1e-6
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11):
        return False
    if not _hexagons_separated(inner_data):
        return False
    if not _contains_in_hex(inner_data, outer_data, side):
        return False
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
    return True


def _compute_min_enclosing_hex(inner_data, angle_deg):
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)

    max_proj = 0.0
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            projs = np.abs(p @ normals.T)
            max_proj = max(max_proj, np.max(projs))

    side = max_proj / (np.sqrt(3) / 2)
    return side


def _smooth_side(inner_data, angle_deg, tau=0.01):
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)

    all_projs = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            projs = np.abs(p @ normals.T)
            all_projs.append(projs)

    all_projs = np.concatenate(all_projs)
    m = np.max(all_projs)
    if m < 1e-10:
        return 0.0
    logsumexp = m + np.log(np.sum(np.exp((all_projs - m) / tau))) * tau
    side = logsumexp / (np.sqrt(3) / 2)
    return side


def _build_inner(r1, r2, gangle, ring2_angles):
    inner = []
    inner.append([0.0, 0.0, gangle])
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    for ang_deg in ring2_angles:
        ang = np.deg2rad(ang_deg)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


def _optimize_configuration(r1_init, r2_init, gangle_init, ring2_angles, deadline):
    def objective(params):
        r1, r2, gangle = params
        if time.time() > deadline:
            return 1e6
        if r1 < 0.5 or r2 < 1.0 or r1 > 10 or r2 > 15:
            return 1000.0
        inner = _build_inner(r1, r2, gangle, ring2_angles)
        outer = np.array([0.0, 0.0, gangle])
        side = _compute_min_enclosing_hex(inner, gangle)
        if not _valid(inner, outer, side):
            return side + 1000.0
        return side

    try:
        res = minimize(
            objective,
            [r1_init, r2_init, gangle_init],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
        )
        if res.fun < 100:
            r1, r2, gangle = res.x
            inner = _build_inner(r1, r2, gangle, ring2_angles)
            outer = np.array([0.0, 0.0, gangle])
            side = _compute_min_enclosing_hex(inner, gangle)
            if _valid(inner, outer, side):
                return inner, outer, side
    except Exception:
        pass

    return None


def _polish_result(inner, outer, side, deadline):
    n_inner = len(inner)

    def polish_objective(params):
        if time.time() > deadline:
            return 1e6
        new_inner = params.reshape(n_inner, 3)
        new_outer = outer.copy()
        new_side = _compute_min_enclosing_hex(new_inner, outer[2])
        if not _valid(new_inner, new_outer, new_side):
            return new_side + 1000.0
        return new_side

    try:
        res = minimize(
            polish_objective,
            inner.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )

        if res.fun < 100:
            new_inner = res.x.reshape(n_inner, 3)
            new_outer = outer.copy()
            new_side = _compute_min_enclosing_hex(new_inner, outer[2])
            if _valid(new_inner, new_outer, new_side) and new_side < side:
                return new_inner, new_outer, new_side
    except Exception:
        pass
    return inner, outer, side


def hexagon_packing_11():
    deadline = time.time() + 580.0

    best_result = None
    best_side = float('inf')

    ring2_configs = [
        [30, 150, 210, 330],
        [0, 120, 180, 300],
        [60, 180, 240, 0],
        [90, 210, 270, 30],
        [150, 270, 330, 90],
        [30, 90, 150, 210],
        [120, 240, 0, 180],
        [60, 120, 180, 240],
    ]

    initial_guesses = [
        [2.5, 4.5, 0.0],
        [2.5, 4.5, 30.0],
        [2.5, 4.5, 60.0],
        [2.0, 4.0, 0.0],
        [3.0, 5.0, 0.0],
        [2.5, 4.5, 15.0],
        [2.5, 4.5, 45.0],
        [2.2, 4.2, 0.0],
        [2.8, 4.8, 30.0],
        [2.5, 4.5, 90.0],
        [2.5, 4.5, 120.0],
    ]

    for ring2_angles in ring2_configs:
        if time.time() > deadline - 30:
            break
        for guess in initial_guesses:
            if time.time() > deadline - 30:
                break
            result = _optimize_configuration(guess[0], guess[1], guess[2], ring2_angles, deadline)
            if result is not None:
                inner, outer, side = result
                if side < best_side:
                    best_side = side
                    best_result = (inner, outer, side)

    if best_result is None or time.time() < deadline - 120:
        for gangle in [0.0, 30.0, 60.0, 15.0, 45.0]:
            for r1 in np.arange(2.0, 3.5, 0.15):
                if time.time() > deadline - 30:
                    break
                for r2 in np.arange(3.5, 6.0, 0.15):
                    for ring2_angles in ring2_configs[:4]:
                        inner = _build_inner(r1, r2, gangle, ring2_angles)
                        outer = np.array([0.0, 0.0, gangle])
                        side = _compute_min_enclosing_hex(inner, gangle)
                        if _valid(inner, outer, side) and side < best_side:
                            best_side = side
                            best_result = (inner, outer, side)

    if best_result is not None:
        inner, outer, side = best_result
        if time.time() < deadline - 20:
            inner, outer, side = _polish_result(inner, outer, side, deadline)
        return inner, outer, side

    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
