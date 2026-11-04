"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_START = time.time()
_EPS = 1e-9


def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0


def _feasible_score(pos, rad):
    """Return (value, feasible). value = sum(rad) if feasible else -inf."""
    n = len(rad)
    if np.any(rad < -_EPS) or np.any(pos[:, 0] - rad < -_EPS) or np.any(pos[:, 0] + rad > 1 + _EPS) \
       or np.any(pos[:, 1] - rad < -_EPS) or np.any(pos[:, 1] + rad > 1 + _EPS):
        return -np.inf, False
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    d2 = dx * dx + dy * dy
    need = rad[:, None] + rad[None, :]
    iu = np.triu_indices(n, 1)
    if np.any(d2[iu] < need[iu] ** 2 - _EPS):
        return -np.inf, False
    return float(np.sum(rad)), True


def _project_feasible(pos, rad, iters=50):
    """Heuristic repair: shrink radii / nudge to satisfy constraints."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    for _ in range(iters):
        wall_changed = False
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            r = min(r, x, 1.0 - x, y, 1.0 - y)
            r = max(r, 0.0)
            x = min(max(x, r), 1.0 - r)
            y = min(max(y, r), 1.0 - r)
            if abs(pos[i, 0] - x) > 1e-12 or abs(pos[i, 1] - y) > 1e-12 or abs(rad[i] - r) > 1e-12:
                wall_changed = True
            pos[i, 0], pos[i, 1], rad[i] = x, y, r
        overlap_changed = False
        for _ in range(5):
            inner_changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - _EPS:
                        s = max((d - _EPS) / need, 0.0) if need > 0 else 1.0
                        rad[i] *= s
                        rad[j] *= s
                        inner_changed = True
            if not inner_changed:
                break
            overlap_changed = True
        if not wall_changed and not overlap_changed:
            break
    val, feas = _feasible_score(pos, rad)
    return pos, rad, feas


def _grow_radii(pos, rad, max_iters=60):
    """Grow radii per-circle iteratively."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    best_rad = rad.copy()
    best_val = float(np.sum(rad))
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                r_from_j = d - rad[j]
                if r_from_j < r_max:
                    r_max = r_from_j
            r_max = max(r_max, 0.0)
            if r_max > rad[i] + 1e-10:
                rad[i] = r_max
                improved = True
        if not improved:
            break
        val = float(np.sum(rad))
        if val > best_val:
            best_val = val
            best_rad = rad.copy()
    return pos, best_rad


def _soft_objective(z, beta):
    """Minimize -sum(r) + beta*(wall + overlap penalties)."""
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    pen += np.sum(np.maximum(r - x, 0.0) ** 2)
    pen += np.sum(np.maximum(x + r - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(r - y, 0.0) ** 2)
    pen += np.sum(np.maximum(y + r - 1.0, 0.0) ** 2)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        pen += np.sum((needu[mask] - du) ** 2)
    return base + beta * pen


def _soft_grad(z, beta):
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    gr -= 1.0
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += beta * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += beta * 2.0 * m[mpos]
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += beta * 2.0 * m[mpos]
        gr[mpos] += beta * 2.0 * m[mpos]
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += beta * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += beta * 2.0 * m[mpos]
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += beta * 2.0 * m[mpos]
        gr[mpos] += beta * 2.0 * m[mpos]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        idx_i = iu[0][mask]
        idx_j = iu[1][mask]
        dxu = dx[iu][mask]
        dyu = dy[iu][mask]
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        fu = needu[mask] - du
        inv_u = 1.0 / du
        np.add.at(gx, idx_i, beta * 2.0 * fu * (-dxu * inv_u))
        np.add.at(gx, idx_j, beta * 2.0 * fu * (dxu * inv_u))
        np.add.at(gy, idx_i, beta * 2.0 * fu * (-dyu * inv_u))
        np.add.at(gy, idx_j, beta * 2.0 * fu * (dyu * inv_u))
        np.add.at(gr, idx_i, beta * 2.0 * fu)
        np.add.at(gr, idx_j, beta * 2.0 * fu)
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def _hex_seed(jitter=0.0, rng=None):
    """Hexagonal arrangement scaled to unit square."""
    rng = rng or np.random.default_rng()
    pts = []
    rows = 0
    while len(pts) < N:
        row = rows
        y = row * 0.8660254
        offset = (0.5 if row % 2 else 0.0)
        col = 0
        while len(pts) < N:
            x = col + offset
            pts.append((x, y))
            col += 1
        rows += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pts = (pts - [xmin, ymin]) / span
    cx_span = (xmax - xmin) / span
    cy_span = (ymax - ymin) / span
    pts[:, 0] = (1.0 - cx_span) / 2 + pts[:, 0] * cx_span
    pts[:, 1] = (1.0 - cy_span) / 2 + pts[:, 1] * cy_span
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.98
    return pts, rad


def _grid_init(rng):
    """Grid-based initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            x = (i + 0.5) / side + rng.normal(0, 0.01)
            y = (j + 0.5) / side + rng.normal(0, 0.01)
            pts.append([x, y])
    pts = np.array(pts[:N])
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad


def _repulsion_init(rng):
    """Random init with repulsion relaxation."""
    pts = rng.uniform(0.1, 0.9, size=(N, 2))
    for it in range(300):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d = np.hypot(dx, dy)
                min_d = 0.1
                if d < min_d:
                    if d < 1e-8:
                        dx, dy = rng.normal(0, 0.01, 2)
                        d = np.hypot(dx, dy)
                    push = (min_d - d) / d * 0.5
                    pts[i, 0] += dx * push
                    pts[i, 1] += dy * push
                    pts[j, 0] -= dx * push
                    pts[j, 1] -= dy * push
                    moved = True
        pts = np.clip(pts, 0.05, 0.95)
        if not moved:
            break
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad


def _langevin_step(pos, rad, dt, temp, rng):
    """One Langevin dynamics step for hard disks. Returns (pos, rad, accepted)."""
    n = len(rad)
    pos_new = pos.copy()
    # Add random kicks
    kicks = rng.normal(0, np.sqrt(2 * temp * dt), size=pos.shape)
    pos_new += kicks
    # Clamp to walls
    for i in range(n):
        pos_new[i, 0] = np.clip(pos_new[i, 0], rad[i], 1.0 - rad[i])
        pos_new[i, 1] = np.clip(pos_new[i, 1], rad[i], 1.0 - rad[i])
    # Check overlaps - if any, reject
    dx = pos_new[:, 0][:, None] - pos_new[:, 0][None, :]
    dy = pos_new[:, 1][:, None] - pos_new[:, 1][None, :]
    d2 = dx * dx + dy * dy
    need = rad[:, None] + rad[None, :]
    iu = np.triu_indices(n, 1)
    if np.any(d2[iu] < need[iu] ** 2 - 1e-10):
        return pos, rad, False
    return pos_new, rad, True


def _langevin_phase(pos, rad, rng, max_steps=5000, start_temp=0.005, end_temp=0.0001):
    """Langevin dynamics to explore configurations."""
    n = len(rad)
    dt = 0.01
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_val = float(np.sum(rad))
    
    for step in range(max_steps):
        frac = step / max_steps
        temp = start_temp * (end_temp / start_temp) ** frac
        # Try to grow radii slightly
        for i in range(n):
            r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            r_max = r_wall
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    r_max = min(r_max, d - rad[j])
            r_max = max(r_max, rad[i])
            rad[i] = min(r_max, rad[i] + 0.0005)
        # Langevin step
        pos, rad, accepted = _langevin_step(pos, rad, dt, temp, rng)
        if not accepted:
            # Try smaller step
            for _ in range(3):
                pos, rad, accepted = _langevin_step(pos, rad, dt * 0.5, temp, rng)
                if accepted:
                    break
        # Check if we improved
        val, feas = _feasible_score(pos, rad)
        if feas and val > best_val:
            best_val = val
            best_pos = pos.copy()
            best_rad = rad.copy()
    return best_pos, best_rad


def _boundary_anchored_init(rng):
    """Initialize circles anchored to boundary features: corners, edges, center."""
    pts = np.zeros((N, 2))
    # 4 corners
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pts[i] = [cx + rng.normal(0, 0.02), cy + rng.normal(0, 0.02)]
    # 4 edge midpoints
    edges = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        pts[4 + i] = [cx + rng.normal(0, 0.02), cy + rng.normal(0, 0.02)]
    # 4 diagonal midpoints
    diags = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    for i, (cx, cy) in enumerate(diags):
        pts[8 + i] = [cx + rng.normal(0, 0.02), cy + rng.normal(0, 0.02)]
    # 4 more edge positions
    extra_edges = [(0.3, 0.08), (0.7, 0.08), (0.3, 0.92), (0.7, 0.92)]
    for i, (cx, cy) in enumerate(extra_edges):
        pts[12 + i] = [cx + rng.normal(0, 0.02), cy + rng.normal(0, 0.02)]
    # 4 more
    extra2 = [(0.08, 0.3), (0.08, 0.7), (0.92, 0.3), (0.92, 0.7)]
    for i, (cx, cy) in enumerate(extra2):
        pts[16 + i] = [cx + rng.normal(0, 0.02), cy + rng.normal(0, 0.02)]
    # 4 center ring
    center_ring = [(0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6)]
    for i, (cx, cy) in enumerate(center_ring):
        pts[20 + i] = [cx + rng.normal(0, 0.01), cy + rng.normal(0, 0.01)]
    # 2 more
    pts[24] = [0.5 + rng.normal(0, 0.01), 0.5 + rng.normal(0, 0.01)]
    pts[25] = [0.5 + rng.normal(0, 0.015), 0.5 + rng.normal(0, 0.015)]
    
    pts = np.clip(pts, 0.05, 0.95)
    
    # Relax with repulsion
    for it in range(200):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d = np.hypot(dx, dy)
                min_d = 0.08
                if d < min_d:
                    if d < 1e-8:
                        dx, dy = rng.normal(0, 0.01, 2)
                        d = np.hypot(dx, dy)
                    push = (min_d - d) / d * 0.5
                    pts[i, 0] += dx * push
                    pts[i, 1] += dy * push
                    pts[j, 0] -= dx * push
                    pts[j, 1] -= dy * push
                    moved = True
        pts = np.clip(pts, 0.05, 0.95)
        if not moved:
            break
    
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.98
    return pts, rad


def _optimize_from_start(pos, rad, beta_schedule, max_iters_per_beta, rng):
    """Run L-BFGS-B with annealing schedule from a given start."""
    bounds = [(0, 1), (0, 1), (1e-5, 1)] * N
    cur_pos, cur_rad = pos.copy(), rad.copy()
    best_local_val = -np.inf
    best_local_pos = None
    best_local_rad = None
    
    for beta in beta_schedule:
        if _remaining() < 8.0:
            break
        z = np.empty(3 * N)
        z[0::3] = cur_pos[:, 0]
        z[1::3] = cur_pos[:, 1]
        z[2::3] = cur_rad
        res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                       bounds=bounds, args=(beta,),
                       options={'maxiter': max_iters_per_beta, 'gtol': 1e-7, 'ftol': 1e-10})
        z = res.x
        cur_pos = np.column_stack([z[0::3], z[1::3]])
        cur_rad = z[2::3]
        cur_pos, cur_rad, feas = _project_feasible(cur_pos, cur_rad, iters=30)
        if feas:
            val = float(np.sum(cur_rad))
            if val > best_local_val:
                best_local_val = val
                best_local_pos = cur_pos.copy()
                best_local_rad = cur_rad.copy()
    
    return best_local_pos, best_local_rad


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(42)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    def _update_best(pos, rad):
        nonlocal best_val, best_pos, best_rad
        val, feas = _feasible_score(pos, rad)
        if feas and val > best_val:
            best_val = val
            best_pos = pos.copy()
            best_rad = rad.copy()

    # Generate diverse initial configurations
    seeds = []
    # Hex seeds with varying jitter
    for jitter in [0.0, 0.01, 0.03, 0.06]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append((p, r))
    # Grid
    p, r = _grid_init(rng)
    seeds.append((p, r))
    # Boundary anchored
    for _ in range(3):
        p, r = _boundary_anchored_init(rng)
        seeds.append((p, r))
    # Repulsion
    for _ in range(4):
        p, r = _repulsion_init(rng)
        seeds.append((p, r))
    # Random
    for _ in range(3):
        p = rng.uniform(0.1, 0.9, size=(N, 2))
        r = np.full(N, 0.0)
        for i in range(N):
            dmin = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
            for j in range(N):
                if i != j:
                    dmin = min(dmin, np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]))
            r[i] = 0.5 * dmin * 0.95
        seeds.append((p, r))

    # Phase 1: Multi-start L-BFGS-B with beta annealing
    beta_schedules = [
        [50.0, 20.0, 8.0, 3.0, 1.0, 0.4, 0.15, 0.05],
        [100.0, 30.0, 10.0, 4.0, 1.5, 0.5, 0.2, 0.08],
    ]
    max_iters_list = [200, 300]

    for si, (pos, rad) in enumerate(seeds):
        if _remaining() < 20.0:
            break
        # Use different schedules for different seeds
        sched_idx = si % len(beta_schedules)
        maxit = max_iters_list[sched_idx]
        bp, br = _optimize_from_start(pos, rad, beta_schedules[sched_idx], maxit, rng)
        if bp is not None:
            _update_best(bp, br)
        # Also try Langevin from this seed if time allows
        if _remaining() > 40.0 and si < 6:
            lp, lr = _langevin_phase(pos, rad, rng, max_steps=3000, start_temp=0.004, end_temp=0.0001)
            _update_best(lp, lr)
            # Polish after Langevin
            if _remaining() > 20.0:
                pp, pr = _optimize_from_start(lp, lr, [5.0, 2.0, 0.5, 0.1], 200, rng)
                if pp is not None:
                    _update_best(pp, pr)

    # Phase 2: Random restarts from best
    if best_pos is not None and _remaining() > 30.0:
        n_restarts = min(8, int(_remaining() // 12))
        for ri in range(n_restarts):
            if _remaining() < 12.0:
                break
            p = best_pos + rng.normal(0, 0.025, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            r = best_rad * rng.uniform(0.6, 1.0, size=N)
            r = np.clip(r, 1e-4, 0.5)
            bp, br = _optimize_from_start(p, r, [20.0, 8.0, 3.0, 1.0, 0.3, 0.1], 250, rng)
            if bp is not None:
                _update_best(bp, br)
            # Grow radii
            if _remaining() > 10.0:
                pg, rg = _grow_radii(bp if bp is not None else p, br if br is not None else r, max_iters=40)
                pg, rg, feas = _project_feasible(pg, rg, iters=15)
                if feas:
                    _update_best(pg, rg)

    # Phase 3: High-precision final polish
    if best_pos is not None and _remaining() > 15.0:
        for beta in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if _remaining() < 8.0:
                break
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=[(0, 1), (0, 1), (1e-5, 1)] * N, args=(beta,),
                           options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=60)
            if feas:
                _update_best(p, r)

        # Final grow
        if _remaining() > 10.0:
            pg, rg = _grow_radii(best_pos, best_rad, max_iters=80)
            pg, rg, feas = _project_feasible(pg, rg, iters=30)
            if feas:
                _update_best(pg, rg)
            # One more polish after grow
            if _remaining() > 8.0:
                z = np.empty(3 * N)
                z[0::3] = best_pos[:, 0]
                z[1::3] = best_pos[:, 1]
                z[2::3] = best_rad
                res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                               bounds=[(0, 1), (0, 1), (1e-5, 1)] * N, args=(0.001,),
                               options={'maxiter': 400, 'gtol': 1e-11, 'ftol': 1e-15})
                z = res.x
                p = np.column_stack([z[0::3], z[1::3]])
                r = z[2::3]
                p, r, feas = _project_feasible(p, r, iters=50)
                if feas:
                    pg2, rg2 = _grow_radii(p, r, max_iters=40)
                    pg2, rg2, feas2 = _project_feasible(pg2, rg2, iters=20)
                    if feas2:
                        _update_best(pg2, rg2)

    # Fallback
    if best_pos is None:
        p, r = _hex_seed(jitter=0.0)
        p, r, feas = _project_feasible(p, r)
        if feas:
            best_pos, best_rad = p, r
        else:
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            arr = np.array(cells[:N])
            return arr

    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
