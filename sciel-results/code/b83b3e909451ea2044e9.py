# EVOLVE-BLOCK-START
import numpy as np
import time

_SQRT3 = np.sqrt(3.0)
_TOL = 1e-6


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _all_vertices(cfg):
    cx = cfg[:, 0]
    cy = cfg[:, 1]
    a = np.radians(cfg[:, 2])
    angles = a[:, None] + np.radians(60 * np.arange(6))[None, :]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    vx = cx[:, None] + cos_a
    vy = cy[:, None] + sin_a
    return np.stack([vx, vy], axis=-1)


def _edge_normals(verts):
    edges = np.roll(verts, -1, axis=1) - verts
    normals = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return normals / norms


def _separated_all(cfg, tol=_TOL):
    verts = _all_vertices(cfg)          # (11,6,2)
    n1 = _edge_normals(verts)           # (11,6,2)
    vmax = np.max(verts, axis=1)        # (11,2)
    vmin = np.min(verts, axis=1)        # (11,2)
    projmax = np.einsum('hce,hc->he', vmax, n1)   # (11,6)
    projmin = np.einsum('hce,hc->he', vmin, n1)   # (11,6)
    H, E = 11, 6
    # normals of hex j: (11,6,2)
    nT = np.transpose(n1, (1, 2, 0))
    projmaxT = np.einsum('jce,je->jc', vmax, nT)  # (6,11)
    projminT = np.einsum('jce,je->jc', vmin, nT)  # (6,11)
    # i normals: projmax[i,e] < projmin[j,e] - tol
    A = projmax[:, :, None] < projminT[None, :, :] - tol   # (11,6,11)
    B = projmaxT[:, None, :] < projmin[None, :, :] - tol   # (6,11,11)
    sep = np.einsum('ije,ije->ij', A, B)                  # (11,11)
    np.fill_diagonal(sep, True)
    return bool(np.all(sep))


def _compute_outer_side(cfg, angle_deg):
    verts = _all_vertices(cfg).reshape(-1, 2)
    a = np.radians(angle_deg)
    theta = a + np.radians(30 + 60 * np.arange(6))
    nx = np.cos(theta)
    ny = np.sin(theta)
    proj = verts[:, 0] * nx + verts[:, 1] * ny
    maxp = np.max(proj)
    minp = np.min(proj)
    if minp > -1e-9:
        minp = 0.0
    return (maxp - minp) / _SQRT3


def _check_containment(cfg, outer_center, outer_angle, side, tol=_TOL):
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    inner_verts = _all_vertices(cfg).reshape(-1, 2)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        cross = ex * (inner_verts[:, 1] - p1[1]) - ey * (inner_verts[:, 0] - p1[0])
        if np.min(cross) < -tol:
            return False
    return True


def _objective(cfg, angle):
    if not _separated_all(cfg):
        return float('inf')
    return _compute_outer_side(cfg, angle)


def _best_angle(cfg, t_end, coarse=0.5, fine=0.02):
    base = 0.0
    best_s = float('inf')
    for ang in np.arange(0.0, 30.0, coarse):
        s = _compute_outer_side(cfg, ang)
        if s < best_s:
            best_s = s
            base = ang
    for ang in np.arange(base - 1.0, base + 1.0 + 1e-9, fine):
        if time.time() > t_end:
            break
        s = _compute_outer_side(cfg, ang)
        if s < best_s:
            best_s = s
            base = ang
    return best_s, base


def _anneal(cfg0, angle, rng, t_end, step0=0.3):
    current = cfg0.copy()
    best = cfg0.copy()
    best_s = _objective(current, angle)
    cur_s = best_s
    step = step0
    start = time.time()
    while time.time() < t_end:
        elapsed = time.time() - start
        T = max(0.005, 0.6 * (1.0 - elapsed / max(1e-9, t_end - start)))
        n_pert = int(rng.integers(1, 5))
        idx = rng.choice(11, size=n_pert, replace=False)
        new = current.copy()
        new[idx, :2] += rng.normal(0.0, step, size=(n_pert, 2))
        if not _separated_all(new):
            step = min(1.0, step * 1.03)
            continue
        s = _compute_outer_side(new, angle)
        if s < cur_s:
            current = new
            cur_s = s
            if s < best_s - 1e-12:
                best_s = s
                best = new.copy()
            step = max(0.01, step * 0.97)
        else:
            if rng.random() < np.exp(-(s - cur_s) / T):
                current = new
                cur_s = s
                step = max(0.01, step * 0.99)
            else:
                step = min(1.0, step * 1.03)
    return best, best_s


def _refine(cfg0, angle, rng, t_end, step0=0.05):
    best = cfg0.copy()
    best_s = _objective(best, angle)
    step = step0
    while time.time() < t_end:
        if rng.random() < 0.35:
            n_upd = int(rng.integers(2, 4))
            idx = rng.choice(11, size=n_upd, replace=False)
            new = best.copy()
            new[idx, :2] += rng.normal(0.0, step * 0.7, size=(n_upd, 2))
        else:
            idx = int(rng.integers(0, 11))
            new = best.copy()
            new[idx, :2] += rng.normal(0.0, step, size=2)
        if not _separated_all(new):
            step = min(0.2, step * 1.06)
            continue
        s = _compute_outer_side(new, angle)
        if s < best_s - 1e-12:
            best_s = s
            best = new.copy()
            step = max(0.001, step * 0.92)
        else:
            step = min(0.2, step * 1.05)
    return best, best_s


def _init_honeycomb():
    dx = 2.0 * _SQRT3
    dy = 1.5 * _SQRT3
    return np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0], [0.5 * dx, dy, 0.0], [-0.5 * dx, dy, 0.0],
        [-dx, 0.0, 0.0], [-0.5 * dx, -dy, 0.0], [0.5 * dx, -dy, 0.0],
        [1.5 * dx, 1.5 * dy, 0.0], [-1.5 * dx, 1.5 * dy, 0.0],
        [1.5 * dx, -1.5 * dy, 0.0], [-1.5 * dx, -1.5 * dy, 0.0],
    ])


def _init_staggered():
    dx = 2.0 * _SQRT3
    dy = 1.5 * _SQRT3
    return np.array([
        [-2.5 * dx, 0.0, 0.0], [-1.5 * dx, 0.0, 0.0], [-0.5 * dx, 0.0, 0.0],
        [0.5 * dx, 0.0, 0.0], [1.5 * dx, 0.0, 0.0], [2.5 * dx, 0.0, 0.0],
        [-2.0 * dx, dy, 0.0], [-1.0 * dx, dy, 0.0], [0.0, dy, 0.0],
        [1.0 * dx, dy, 0.0], [2.0 * dx, dy, 0.0],
    ])


def _init_compact():
    dx = 2.0 * _SQRT3
    dy = 1.5 * _SQRT3
    return np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0], [0.5 * dx, dy, 0.0], [-0.5 * dx, dy, 0.0],
        [-dx, 0.0, 0.0], [-0.5 * dx, -dy, 0.0], [0.5 * dx, -dy, 0.0],
        [1.5 * dx, 1.5 * dy, 0.0], [-1.5 * dx, 1.5 * dy, 0.0],
        [0.0, 3.0 * dy, 0.0], [0.0, -3.0 * dy, 0.0],
    ])


def _init_ring(rng, r1=None, r2=None):
    config = np.zeros((11, 3))
    config[0] = [0.0, 0.0, 0.0]
    if r1 is None:
        r1 = 2.0 * _SQRT3 + rng.normal(0.0, 0.15)
    for i in range(6):
        a = 2 * np.pi * i / 6 + rng.normal(0.0, 0.1)
        config[1 + i] = [r1 * np.cos(a), r1 * np.sin(a), 0.0]
    if r2 is None:
        r2 = r1 * (1.4 + rng.uniform(-0.1, 0.3))
    pattern = int(rng.integers(0, 4))
    if pattern == 0:
        oa = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    elif pattern == 1:
        oa = [0.0, np.pi / 3, np.pi, 4 * np.pi / 3]
    elif pattern == 2:
        oa = [rng.uniform(0, np.pi / 6), rng.uniform(np.pi / 2, 2 * np.pi / 3),
              rng.uniform(np.pi, 7 * np.pi / 6), rng.uniform(4 * np.pi / 3, 5 * np.pi / 3)]
    else:
        oa = [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6]
    for i in range(4):
        a = oa[i]
        rr = r2 + rng.normal(0.0, 0.1)
        config[7 + i] = [rr * np.cos(a), rr * np.sin(a), 0.0]
    return config


def _init_compact_sat(rng):
    dx = 2.0 * _SQRT3
    dy = 1.5 * _SQRT3
    config = np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0], [0.5 * dx, dy, 0.0], [-0.5 * dx, dy, 0.0],
        [-dx, 0.0, 0.0], [-0.5 * dx, -dy, 0.0], [0.5 * dx, -dy, 0.0],
    ])
    sats = [
        [1.5 * dx + rng.normal(0, 0.3), 0.5 * dy + rng.normal(0, 0.3)],
        [-1.5 * dx + rng.normal(0, 0.3), 0.5 * dy + rng.normal(0, 0.3)],
        [0.5 * dx + rng.normal(0, 0.3), 2.0 * dy + rng.normal(0, 0.3)],
        [-0.5 * dx + rng.normal(0, 0.3), -2.0 * dy + rng.normal(0, 0.3)],
    ]
    for sx, sy in sats:
        config = np.vstack([config, [[sx, sy, 0.0]]])
    return config


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 575.0
    rng = np.random.default_rng(42)

    best_cfg = None
    best_side = float('inf')
    best_angle = 0.0

    def consider(cfg, t_end):
        nonlocal best_cfg, best_side, best_angle
        if not _separated_all(cfg):
            return
        s, a = _best_angle(cfg, t_end)
        if s < best_side:
            best_side = s
            best_cfg = cfg.copy()
            best_angle = a

    # Build diverse initial pool
    inits = [_init_honeycomb(), _init_staggered(), _init_compact()]
    for _ in range(6):
        inits.append(_init_ring(rng))
    for _ in range(4):
        inits.append(_init_compact_sat(rng))
    base = _init_honeycomb()
    for _ in range(3):
        p = base.copy()
        p[:, :2] += rng.normal(0.0, 0.4, size=(11, 2))
        inits.append(p)

    # Score all inits
    for cfg in inits:
        if time.time() > deadline - 150.0:
            break
        consider(cfg, deadline - 150.0)

    # Main optimization loop over the best config, alternating anneal/refine
    guard = 0
    while time.time() < deadline - 60.0 and best_cfg is not None:
        guard += 1
        remaining = deadline - time.time()
        if remaining < 25.0:
            break
        t_budget = min(remaining * 0.5, 90.0)
        if guard % 2 == 0:
            cfg2, s2 = _anneal(best_cfg, best_angle, rng, time.time() + t_budget, step0=0.25)
        else:
            cfg2, s2 = _refine(best_cfg, best_angle, rng, time.time() + t_budget, step0=0.05)
        if s2 < best_side:
            best_side = s2
            best_cfg = cfg2
        # occasionally re-derive the best angle
        if time.time() < deadline - 30.0:
            s, a = _best_angle(best_cfg, deadline - 30.0, coarse=0.25, fine=0.01)
            if s < best_side:
                best_side = s
                best_angle = a

    # Final fine angle sweep
    if best_cfg is not None:
        for ang in np.arange(best_angle - 1.0, best_angle + 1.0 + 1e-9, 0.01):
            s = _compute_outer_side(best_cfg, ang)
            if s < best_side:
                best_side = s
                best_angle = ang

    outer_hex_data = np.array([0.0, 0.0, best_angle])

    if best_cfg is None:
        best_cfg = _init_honeycomb()
        best_side, best_angle = _best_angle(best_cfg, deadline)
        outer_hex_data = np.array([0.0, 0.0, best_angle])

    # Ensure containment (grow side only if needed)
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
        for _ in range(5000):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                break
            best_side *= 1.0005

    return best_cfg, outer_hex_data, best_side


# EVOLVE-BLOCK-END
