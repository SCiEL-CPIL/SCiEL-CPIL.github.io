"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 148.0
_START = time.time()
_EPS = 1e-9


def _remaining():
    return _DEADLINE - (time.time() - _START) - 3.0


def _feasible_score(pos, rad):
    """Return (value, feasible). value = sum(rad) if feasible else -inf."""
    n = len(rad)
    if np.any(rad < -_EPS) or np.any(pos[:, 0] - rad < -_EPS) or np.any(pos[:, 0] + rad > 1 + _EPS) \
       or np.any(pos[:, 1] - rad < -_EPS) or np.any(pos[:, 1] + rad > 1 + _EPS):
        return -np.inf, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d2 = dx * dx + dy * dy
            rr = rad[i] + rad[j]
            if d2 < rr * rr - _EPS:
                return -np.inf, False
    return float(np.sum(rad)), True


def _project_feasible(pos, rad, iters=60):
    """Heuristic repair: shrink radii / nudge to satisfy constraints."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    for _ in range(iters):
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            r = min(r, x, 1.0 - x, y, 1.0 - y)
            r = max(r, 0.0)
            x = min(max(x, r), 1.0 - r)
            y = min(max(y, r), 1.0 - r)
            pos[i, 0], pos[i, 1], rad[i] = x, y, r
        changed = False
        for _ in range(3):
            inner_changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - _EPS:
                        a = rad[i]
                        b = rad[j]
                        s = (d - _EPS) / (need + _EPS) if need > 0 else 1.0
                        rad[i] = max(a * s, 0.0)
                        rad[j] = max(b * s, 0.0)
                        inner_changed = True
            if not inner_changed:
                break
            changed = True
        if not changed:
            break
    val, feas = _feasible_score(pos, rad)
    return pos, rad, feas


def _soft_objective(z, mu):
    """
    z = [x0,y0,r0, x1,y1,r1, ...]
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty)
    Fully vectorized.
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    obj = -np.sum(r)

    # Boundary violations
    b_left = np.maximum(0.0, r - x)
    b_bot = np.maximum(0.0, r - y)
    b_right = np.maximum(0.0, r - (1.0 - x))
    b_top = np.maximum(0.0, r - (1.0 - y))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    iu = np.triu_indices(n, 1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-24))
    rs_sum = r[iu[0]] + r[iu[1]]
    ov = np.maximum(0.0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    return obj + mu * (bpen + open_)


def _soft_grad(z, mu):
    """Gradient of _soft_objective. Fully vectorized."""
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]

    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n) - 1.0  # d(-sum r)/dr = -1

    # Boundary penalty gradient
    b_left = np.maximum(0.0, r - x)
    b_bot = np.maximum(0.0, r - y)
    b_right = np.maximum(0.0, r - (1.0 - x))
    b_top = np.maximum(0.0, r - (1.0 - y))
    bv = b_left + b_bot + b_right + b_top

    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    gx += mu * 2 * bv * db_dxi
    gy += mu * 2 * bv * db_dyi
    gr += mu * 2 * bv * db_dri

    # Overlap penalty gradient
    iu = np.triu_indices(n, 1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-24))
    rs_sum = r[iu[0]] + r[iu[1]]
    ov = np.maximum(0.0, rs_sum - dist)

    mask = ov > 0
    if np.any(mask):
        im = iu[0][mask]
        jm = iu[1][mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        np.add.at(gx, im, -mu * 2 * ov_m * ux)
        np.add.at(gx, jm, mu * 2 * ov_m * ux)
        np.add.at(gy, im, -mu * 2 * ov_m * uy)
        np.add.at(gy, jm, mu * 2 * ov_m * uy)
        np.add.at(gr, im, mu * 2 * ov_m)
        np.add.at(gr, jm, mu * 2 * ov_m)

    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def _epigraph_objective(z, beta):
    """
    Epigraph formulation with smooth max-gap surrogate.
    z = [x0,y0,r0, ..., x_{n-1},y_{n-1},r_{n-1}, t]
    t represents the minimum clearance (distance between circle boundaries and walls/other circles).
    We maximize sum(r) + beta * t, where t <= min of all clearances.
    
    Clearances:
    - Wall: x_i - r_i, 1-x_i-r_i, y_i-r_i, 1-y_i-r_i
    - Pair: dist(i,j) - r_i - r_j
    
    Smooth max: t = min_i c_i, approximated by -log-sum-exp(-beta*c_i)/beta
    Actually we want t <= all c_i, and we maximize t, so t = min(c_i).
    Smooth min: softmin(c) = -(1/beta)*log(sum(exp(-beta*c_i)))
    
    Objective: -sum(r) - beta * softmin(clearances)
    This encourages all clearances to be large (uniformly).
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    t = z[-1]  # auxiliary variable

    # All clearances
    wall_clear = np.concatenate([
        x - r,       # left
        1.0 - x - r,  # right
        y - r,       # bottom
        1.0 - y - r   # top
    ])

    iu = np.triu_indices(n, 1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-24))
    pair_clear = dist - r[iu[0]] - r[iu[1]]

    all_clear = np.concatenate([wall_clear, pair_clear])

    # Soft min: softmin(c) = -(1/beta) * log(sum(exp(-beta * c)))
    # For numerical stability, shift by max
    c_shifted = all_clear - np.max(all_clear)
    log_sum = np.log(np.sum(np.exp(-beta * c_shifted)))
    softmin = -(1.0 / beta) * log_sum + np.max(all_clear)

    # Objective: minimize -sum(r) - softmin (we want to maximize sum(r) + softmin)
    # But we also need t <= softmin, so penalty if t > softmin
    # Actually, let's just use: minimize -sum(r) - softmin
    # The t variable is implicit; we don't need it explicitly if we use softmin directly.
    
    # Simpler: just minimize -sum(r) - softmin(clearances)
    return -np.sum(r) - softmin


def _epigraph_grad(z, beta):
    """Gradient of _epigraph_objective."""
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]

    # Wall clearances
    c_left = x - r
    c_right = 1.0 - x - r
    c_bot = y - r
    c_top = 1.0 - y - r

    iu = np.triu_indices(n, 1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-24))
    c_pair = dist - r[iu[0]] - r[iu[1]]

    all_clear = np.concatenate([c_left, c_right, c_bot, c_top, c_pair])

    # Soft min gradient: d(softmin)/dc_i = exp(-beta*c_i) / sum(exp(-beta*c_j))
    c_shifted = all_clear - np.max(all_clear)
    exp_neg = np.exp(-beta * c_shifted)
    weights = exp_neg / np.sum(exp_neg)

    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n) - 1.0  # -sum(r)

    # Wall gradients: d(c)/d(x,y,r)
    # c_left = x - r: dgx=+1, dgr=-1
    # c_right = 1-x-r: dgx=-1, dgr=-1
    # c_bot = y - r: dgy=+1, dgr=-1
    # c_top = 1-y-r: dgy=-1, dgr=-1
    w_left = weights[:n]
    w_right = weights[n:2*n]
    w_bot = weights[2*n:3*n]
    w_top = weights[3*n:4*n]

    gx += w_left - w_right
    gy += w_bot - w_top
    gr += -(w_left + w_right + w_bot + w_top)

    # Pair gradients
    w_pair = weights[4*n:]
    safe_dist = np.where(dist < 1e-12, 1.0, dist)
    ux = dx / safe_dist
    uy = dy / safe_dist

    np.add.at(gx, iu[0], w_pair * ux)
    np.add.at(gx, iu[1], -w_pair * ux)
    np.add.at(gy, iu[0], w_pair * uy)
    np.add.at(gy, iu[1], -w_pair * uy)
    np.add.at(gr, iu[0], -w_pair)
    np.add.at(gr, iu[1], -w_pair)

    out = np.empty(3 * n + 1)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    out[-1] = 0.0  # t is implicit, no gradient needed
    return out


def _hex_seed(jitter=0.0, rng=None):
    """Generate hexagonal arrangement."""
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
    cx = (xmax - xmin) / span
    cy = (ymax - ymin) / span
    pts[:, 0] = 0.5 - cx / 2 + pts[:, 0] * cx
    pts[:, 1] = 0.5 - cy / 2 + pts[:, 1] * cy
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    rad = np.full(N, 0.2)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.98
    return pts, rad


def _langevin_dynamics(pos, rad, rng, n_cool=2000):
    """Langevin dynamics for hard disks with adaptive cooling."""
    n = len(rad)
    pos = pos.astype(np.float64).copy()
    vel = np.zeros((n, 2))
    vel += rng.normal(0, 0.05, size=(n, 2))
    T0 = 0.05
    Tend = 0.0005
    dt = 0.002

    for step in range(n_cool):
        if _remaining() < 5.0:
            break
        alpha = step / max(n_cool - 1, 1)
        T = T0 * (Tend / T0) ** alpha

        force = np.zeros((n, 2))
        wall_k = 50.0
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x < r:
                force[i, 0] += wall_k * (r - x)
            if x > 1.0 - r:
                force[i, 0] -= wall_k * (x - (1.0 - r))
            if y < r:
                force[i, 1] += wall_k * (r - y)
            if y > 1.0 - r:
                force[i, 1] -= wall_k * (y - (1.0 - r))

        pair_k = 100.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                need = rad[i] + rad[j]
                if d < need and d > 1e-10:
                    overlap = need - d
                    f = pair_k * overlap
                    fx = f * dx / d
                    fy = f * dy / d
                    force[i, 0] += fx
                    force[i, 1] += fy
                    force[j, 0] -= fx
                    force[j, 1] -= fy
                elif d <= 1e-10:
                    angle = rng.uniform(0, 2 * np.pi)
                    force[i, 0] += pair_k * 0.01 * np.cos(angle)
                    force[i, 1] += pair_k * 0.01 * np.sin(angle)
                    force[j, 0] -= pair_k * 0.01 * np.cos(angle)
                    force[j, 1] -= pair_k * 0.01 * np.sin(angle)

        damping = 0.95
        noise_amp = np.sqrt(2 * T * dt)
        vel = vel * damping + force * dt + rng.normal(0, noise_amp, size=(n, 2))
        pos += vel * dt

        for i in range(n):
            r = rad[i]
            if pos[i, 0] < r:
                pos[i, 0] = r
                vel[i, 0] = abs(vel[i, 0]) * 0.5
            if pos[i, 0] > 1.0 - r:
                pos[i, 0] = 1.0 - r
                vel[i, 0] = -abs(vel[i, 0]) * 0.5
            if pos[i, 1] < r:
                pos[i, 1] = r
                vel[i, 1] = abs(vel[i, 1]) * 0.5
            if pos[i, 1] > 1.0 - r:
                pos[i, 1] = 1.0 - r
                vel[i, 1] = -abs(vel[i, 1]) * 0.5

    return pos, rad


def _grow_radii(pos, rad, rng, max_grow=400, growth_step=0.001):
    """Gradually grow all radii while maintaining feasibility."""
    n = len(rad)
    pos = pos.copy()
    rad = rad.copy()

    for _ in range(max_grow):
        if _remaining() < 5.0:
            break
        rad[:] += growth_step

        for repair_iter in range(40):
            has_overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - 1e-10:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            for i in range(n):
                if pos[i, 0] < rad[i] - 1e-10 or pos[i, 0] > 1.0 - rad[i] + 1e-10 or \
                   pos[i, 1] < rad[i] - 1e-10 or pos[i, 1] > 1.0 - rad[i] + 1e-10:
                    has_overlap = True
                    break

            if not has_overlap:
                break

            force = np.zeros((n, 2))
            wall_k = 50.0
            for i in range(n):
                x, y, r = pos[i, 0], pos[i, 1], rad[i]
                if x < r:
                    force[i, 0] += wall_k * (r - x)
                if x > 1.0 - r:
                    force[i, 0] -= wall_k * (x - (1.0 - r))
                if y < r:
                    force[i, 1] += wall_k * (r - y)
                if y > 1.0 - r:
                    force[i, 1] -= wall_k * (y - (1.0 - r))

            pair_k = 200.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need and d > 1e-10:
                        overlap = need - d
                        f = pair_k * overlap
                        fx = f * dx / d
                        fy = f * dy / d
                        force[i, 0] += fx
                        force[i, 1] += fy
                        force[j, 0] -= fx
                        force[j, 1] -= fy

            dt = 0.001
            pos += force * dt
            for i in range(n):
                r = rad[i]
                pos[i, 0] = np.clip(pos[i, 0], r, 1.0 - r)
                pos[i, 1] = np.clip(pos[i, 1], r, 1.0 - r)

        val, feas = _feasible_score(pos, rad)
        if not feas:
            rad[:] -= growth_step
            growth_step *= 0.9
            if growth_step < 1e-5:
                break

    return pos, rad, _feasible_score(pos, rad)[1]


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(12345)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    # Seed set
    seeds = []
    for jitter in [0.0, 0.02, 0.05, 0.1]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append((p, r))
    for _ in range(8):
        p = rng.uniform(0.15, 0.85, size=(N, 2))
        r = rng.uniform(0.05, 0.12, size=N)
        seeds.append((p, r))

    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N

    def _try_anneal(pos, rad, lams, max_iter_each=250):
        nonlocal best_val, best_pos, best_rad
        cur_pos, cur_rad = pos.copy(), rad.copy()
        for lam in lams:
            if _remaining() < 8.0:
                break
            z = np.empty(3 * N)
            z[0::3] = cur_pos[:, 0]
            z[1::3] = cur_pos[:, 1]
            z[2::3] = cur_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': max_iter_each, 'gtol': 1e-7, 'ftol': 1e-9})
            z = res.x
            cur_pos = np.column_stack([z[0::3], z[1::3]])
            cur_rad = z[2::3]
            cur_pos, cur_rad, feas = _project_feasible(cur_pos, cur_rad)
            if feas:
                val = float(np.sum(cur_rad))
                if val > best_val:
                    best_val = val
                    best_pos = cur_pos.copy()
                    best_rad = cur_rad.copy()
        return cur_pos, cur_rad

    # Phase 1: Annealed optimization from multiple seeds
    lams_coarse = [50.0, 20.0, 8.0, 3.0, 1.0, 0.4, 0.15]
    for si, (pos, rad) in enumerate(seeds):
        if _remaining() < 15.0:
            break
        _try_anneal(pos, rad, lams_coarse, max_iter_each=200)

    # Phase 2: Langevin dynamics exploration
    if best_pos is not None and _remaining() > 35.0:
        for ld_trial in range(10):
            if _remaining() < 12.0:
                break
            p = best_pos + rng.normal(0, 0.05, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            r = best_rad * rng.uniform(0.75, 1.0, size=N)
            p_ld, r_ld = _langevin_dynamics(p, r, rng, n_cool=1500)
            p_ld, r_ld, feas = _project_feasible(p_ld, r_ld)
            if feas:
                val = float(np.sum(r_ld))
                if val > best_val:
                    best_val = val
                    best_pos = p_ld.copy()
                    best_rad = r_ld.copy()
            # Grow radii from Langevin result
            if feas and _remaining() > 10.0:
                p_g, r_g, feas_g = _grow_radii(p_ld, r_ld, rng, max_grow=300, growth_step=0.0008)
                if feas_g:
                    val_g = float(np.sum(r_g))
                    if val_g > best_val:
                        best_val = val_g
                        best_pos = p_g.copy()
                        best_rad = r_g.copy()

    # Phase 3: Epigraph-based polish (smooth max-gap optimization)
    if best_pos is not None and _remaining() > 20.0:
        # Use epigraph formulation for final polish
        for beta in [5.0, 10.0, 20.0, 50.0]:
            if _remaining() < 8.0:
                break
            z = np.empty(3 * N + 1)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            z[-1] = 0.0  # t init

            # Bounds for epigraph: x,y in [0,1], r in [1e-4, 1], t in [-1, 1]
            ep_bounds = bounds + [(-1.0, 1.0)]

            res = minimize(_epigraph_objective, z, jac=_epigraph_grad, method='L-BFGS-B',
                           bounds=ep_bounds, args=(beta,),
                           options={'maxiter': 400, 'gtol': 1e-9, 'ftol': 1e-12})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=80)
            if feas:
                val = float(np.sum(r))
                if val > best_val:
                    best_val = val
                    best_pos = p.copy()
                    best_rad = r.copy()

    # Phase 4: More random restarts with quick anneal
    if best_pos is not None and _remaining() > 20.0:
        for _ in range(12):
            if _remaining() < 8.0:
                break
            p = best_pos + rng.normal(0, 0.03, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            r = best_rad * rng.uniform(0.7, 1.0, size=N)
            _try_anneal(p, r, [10.0, 3.0, 0.5, 0.1], max_iter_each=150)

    # Phase 5: Final high-precision polish
    if best_pos is not None and _remaining() > 12.0:
        for lam in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if _remaining() < 6.0:
                break
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=100)
            if feas:
                val = float(np.sum(r))
                if val > best_val:
                    best_val = val
                    best_pos = p.copy()
                    best_rad = r.copy()

        # One more epigraph pass with high beta
        if _remaining() > 6.0:
            z = np.empty(3 * N + 1)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            z[-1] = 0.0
            ep_bounds = bounds + [(-1.0, 1.0)]
            res = minimize(_epigraph_objective, z, jac=_epigraph_grad, method='L-BFGS-B',
                           bounds=ep_bounds, args=(100.0,),
                           options={'maxiter': 600, 'gtol': 1e-10, 'ftol': 1e-14})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=100)
            if feas:
                val = float(np.sum(r))
                if val > best_val:
                    best_val = val
                    best_pos = p.copy()
                    best_rad = r.copy()

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
        best_pos, best_rad = p, r

    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
