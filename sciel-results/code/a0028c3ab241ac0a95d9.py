"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

STRATEGY: Vectorized simulated annealing over all 3*n free variables using a
smooth soft-constraint objective (log-sum-exp penalty), followed by a
vectorized best-fit local refinement. This is a stochastic global-search
paradigm, structurally different from the contact-graph / gradient approach.
"""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """Return (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _soft_obj(v, n, beta):
    """Smooth objective: -sum(r) + penalty. Vectorized, no Python loops."""
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    # Wall violations: r - min(x,1-x,y,1-y)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Circle-circle violations (vectorized over all pairs)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    viol = r[:, None] + r[None, :] - dist
    np.fill_diagonal(viol, 0.0)
    # only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_viol = viol[iu]
    pair_pen = np.sum(np.logaddexp(0.0, beta * pair_viol)) / beta

    return -np.sum(r) + 100.0 * (wall_pen + pair_pen)


def _sa_packing(n, deadline, seed, t0, t1, steps):
    """Vectorized simulated annealing. Returns (xy, r)."""
    rng = np.random.RandomState(seed)

    def _init():
        # hexagonal-ish random init
        x = rng.uniform(0.05, 0.95, n)
        y = rng.uniform(0.05, 0.95, n)
        r = rng.uniform(0.02, 0.05, n)
        return np.concatenate([x, y, r])

    v = _init()
    beta = 50.0
    cur = _soft_obj(v, n, beta)
    best_v = v.copy()
    best_cur = cur

    step_scale = 0.05
    for it in range(steps):
        if time.time() > deadline:
            break
        temp = t0 * (t1 / t0) ** (it / max(1, steps - 1))

        # propose perturbation: random walk in all 3n dims
        dv = rng.normal(0.0, step_scale, 3 * n)
        v_new = v + dv
        # keep radii positive-ish
        v_new[2 * n:3 * n] = np.maximum(v_new[2 * n:3 * n], 0.001)

        new = _soft_obj(v_new, n, beta)
        if new < cur or rng.rand() < np.exp(-(new - cur) / max(temp, 1e-9)):
            v = v_new
            cur = new
            if cur < best_cur:
                best_cur = cur
                best_v = v.copy()
        # slowly increase beta to tighten constraints
        beta = min(200.0, beta * 1.0005)

    return np.column_stack([best_v[0:n], best_v[n:2 * n]]), best_v[2 * n:3 * n]


def _grid_init(side):
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _hex_init():
    """Hexagonal lattice init."""
    positions = []
    spacing = 0.18
    row = 0
    xoff = 0.0
    while len(positions) < N:
        y = 0.1 + row * spacing * 0.866
        if y > 0.95:
            break
        col = 0
        xoff = (spacing / 2) if (row % 2) else 0.0
        while True:
            x = 0.1 + xoff + col * spacing
            if x > 0.95:
                break
            positions.append([x, y, spacing * 0.48])
            col += 1
        row += 1
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _local_search_refine(xy, r, deadline, grid_size=70):
    """Vectorized best-fit local search."""
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 12
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _gradient_polish(xy, r, deadline):
    """L-BFGS-B polish with vectorized analytic gradient."""
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    bounds = []
    for _ in range(n):
        bounds.append((0.001, 0.999))
    for _ in range(n):
        bounds.append((0.001, 0.999))
    for _ in range(n):
        bounds.append((0.001, 0.5))

    beta = 100.0

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _soft_obj(v, n, beta)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        x = v[0:n]
        y = v[n:2 * n]
        r = v[2 * n:3 * n]
        g = np.zeros(3 * n)

        # Wall penalty gradient
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        wexp = np.exp(beta * wall_viol) / (1.0 + np.exp(beta * wall_viol))
        # d wall_pen / d r = wexp
        g[2 * n:3 * n] += 100.0 * wexp
        # d wall_pen / d x, d y: -wexp * (which wall is min)
        # compute subgradient: for each i, the active wall
        for i in range(n):
            wx = np.minimum(x[i], 1.0 - x[i])
            wy = np.minimum(y[i], 1.0 - y[i])
            if wx < wy:
                if x[i] < 0.5:
                    g[i] -= 100.0 * wexp[i]
                else:
                    g[i] += 100.0 * wexp[i]
            else:
                if y[i] < 0.5:
                    g[n + i] -= 100.0 * wexp[i]
                else:
                    g[n + i] += 100.0 * wexp[i]

        # Pair penalty gradient (vectorized with np.add.at)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        viol = r[:, None] + r[None, :] - dist
        np.fill_diagonal(viol, 0.0)
        pexp = np.exp(beta * viol) / (1.0 + np.exp(beta * viol))
        np.fill_diagonal(pexp, 0.0)

        # d pair_pen / d r_i = sum_j pexp_ij
        g[2 * n:3 * n] += 100.0 * np.sum(pexp, axis=1)
        # d pair_pen / d x_i = sum_j pexp_ij * (x_i - x_j)/dist_ij
        gx = pexp * (dx / dist)
        gy = pexp * (dy / dist)
        np.fill_diagonal(gx, 0.0)
        np.fill_diagonal(gy, 0.0)
        g[0:n] += 100.0 * np.sum(gx, axis=1)
        g[n:2 * n] += 100.0 * np.sum(gy, axis=1)

        # objective -sum(r)
        g[2 * n:3 * n] -= 1.0
        return g

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 20000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2 * n]])
        new_r = v_opt[2 * n:3 * n]
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    return xy, r


def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    r = rng.uniform(0.02, 0.05, N)
    return np.column_stack([x, y]), r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    def _consider(xy, r):
        nonlocal best_xy, best_r, best_total, best_valid
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    # Phase 1: SA from multiple diverse initializations
    # hex + grid + several random seeds
    inits = []
    hex_xy = _hex_init()
    inits.append((hex_xy[:, :2].copy(), hex_xy[:, 2].copy(), 101))
    for side in [5, 6]:
        g = _grid_init(side)
        inits.append((g[:, :2].copy(), g[:, 2].copy(), 200 + side))
    for s in range(9):
        rx, rr = _random_init(1000 + s * 7)
        inits.append((rx, rr, 300 + s * 13))

    # distribute SA steps across inits within budget
    n_inits = len(inits)
    for k, (xy0, r0, seed) in enumerate(inits):
        if time.time() > deadline - 40:
            break
        per_init_deadline = min(deadline - 40, start_time + (40.0 / n_inits) * (k + 1))
        steps = int(12000 * (40.0 / n_inits))
        xy, r = _sa_packing(N, per_init_deadline, seed, t0=0.5, t1=0.001, steps=steps)
        # quick local refine
        lxy, lr, lt, lv = _local_search_refine(xy, r, deadline, grid_size=50)
        _consider(lxy, lr)

    # Phase 2: Gradient polish on best
    if best_valid and best_xy is not None and time.time() < deadline - 20:
        pxy, pr = _gradient_polish(best_xy, best_r, deadline)
        _consider(pxy, pr)

    # Phase 3: Final high-res local search
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        lxy, lr, lt, lv = _local_search_refine(best_xy.copy(), best_r.copy(),
                                               deadline, grid_size=90)
        _consider(lxy, lr)

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        g = _grid_init(side)
        _consider(g[:, :2].copy(), g[:, 2].copy())

    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
