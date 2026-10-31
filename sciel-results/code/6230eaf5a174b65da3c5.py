"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# Precompute pair indices
_II, _JJ = np.triu_indices(N, k=1)
_PAIR_MASK = np.triu(np.ones((N, N), dtype=bool), k=1)

# ---------------------------------------------------------------------------
# Smooth (softplus) objective  -- from inspiration; better-conditioned than
# hard max(0,v)^2, so L-BFGS-B gradients are reliable and the optimizer can
# push radii toward their true geometric limits.
# ---------------------------------------------------------------------------
def _soft_max(v, beta):
    """Smooth approximation of max(0, v) via softplus."""
    v = np.asarray(v, dtype=float)
    return np.where(v > 20.0 / beta, v, np.log1p(np.exp(beta * v)) / beta)


def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Boundary violations (smooth)
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] + radii - 1.0
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] + radii - 1.0
    boundary_pen = (np.sum(_soft_max(left_v, beta)) + np.sum(_soft_max(right_v, beta)) +
                    np.sum(_soft_max(bottom_v, beta)) + np.sum(_soft_max(top_v, beta)))

    # Overlap violations (smooth, vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    viol = min_dist * min_dist - dist_sq
    overlap_pen = np.sum(_soft_max(viol[_PAIR_MASK], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)


# ---------------------------------------------------------------------------
# Validity + repair
# ---------------------------------------------------------------------------
def _is_valid(pos, radii, tol=1e-7):
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[_II] + radii[_JJ]
    if np.any(dist_sq < (min_dist - tol) ** 2):
        return False
    return True


def _post_process(pos, radii):
    """Shrink radii minimally to guarantee strict validity (positions fixed)."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(300):
        changed = False
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(max_r * (1.0 - 1e-10), 1e-6)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                s = radii[i] + radii[j]
                if s > 0 and dist < s:
                    k = dist / s * (1.0 - 1e-10)
                    radii[i] = max(radii[i] * k, 1e-6)
                    radii[j] = max(radii[j] * k, 1e-6)
                    changed = True
        if not changed:
            break
    return pos, radii


# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------
def _best_fit_init(seed=0, center_first=True):
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    if center_first:
        pos[0] = [0.5, 0.5]
        radii[0] = 0.249
    else:
        pos[0] = [0.15, 0.15]
        radii[0] = 0.149

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(15000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                pr = d - radii[j]
                if pr < max_r:
                    max_r = pr
            if max_r > best_r:
                best_r, best_x, best_y = max_r, x, y
        if best_r > 0.004:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.97
        else:
            for _ in range(2000):
                x = rng.uniform(0.02, 0.98)
                y = rng.uniform(0.02, 0.98)
                min_d = min(x, 1 - x, y, 1 - y)
                for j in range(i):
                    d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                    min_d = min(min_d, d - radii[j])
                if min_d > 0.004:
                    pos[i] = [x, y]
                    radii[i] = min_d * 0.95
                    break
            else:
                pos[i] = [0.5, 0.5]
                radii[i] = 0.002
    return pos, radii


def _hex_init():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    rng = np.random.RandomState(123)
    while k < N:
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(5000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                max_r = min(max_r, d - radii[j])
            if max_r > best_r:
                best_r, best_x, best_y = max_r, x, y
        if best_r > 0.01:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.98
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        k += 1
    return pos, radii


# ---------------------------------------------------------------------------
# Projected gradient ascent polish (final high-precision pass)
# ---------------------------------------------------------------------------
def _projected_polish(pos, radii, deadline, steps=20000):
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    rng = np.random.RandomState(7)
    step = 0.004
    it = 0
    while it < steps:
        if time.time() > deadline - 3:
            break
        it += 1
        dx = np.zeros((N, 2))
        dr = np.zeros(N)
        for i in range(N):
            # wall gradient
            dx[i, 0] = 1.0 if pos[i, 0] < radii[i] else (-1.0 if pos[i, 0] > 1.0 - radii[i] else 0.0)
            dx[i, 1] = 1.0 if pos[i, 1] < radii[i] else (-1.0 if pos[i, 1] > 1.0 - radii[i] else 0.0)
            # overlap gradient
            for j in range(N):
                if i == j:
                    continue
                vx = pos[i, 0] - pos[j, 0]
                vy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(vx * vx + vy * vy) + 1e-12
                if d < radii[i] + radii[j]:
                    dx[i, 0] += vx / d
                    dx[i, 1] += vy / d
            # radius gradient: 1 minus overlap pressure
            g = 1.0
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2) + 1e-12
                if d < radii[i] + radii[j]:
                    g -= 1.0
            dr[i] = g
        pos = pos + step * dx
        radii = radii + step * dr
        # project
        for i in range(N):
            radii[i] = max(min(radii[i], min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])), 1e-6)
        for i in range(N):
            for j in range(i + 1, N):
                vx = pos[i, 0] - pos[j, 0]
                vy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(vx * vx + vy * vy)
                s = radii[i] + radii[j]
                if s > 0 and d < s:
                    k = d / s * (1.0 - 1e-9)
                    radii[i] = max(radii[i] * k, 1e-6)
                    radii[j] = max(radii[j] * k, 1e-6)
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
        else:
            step *= 0.99
            if rng.random() < 0.05:
                idx = rng.randint(N)
                pos[idx, 0] += rng.uniform(-0.01, 0.01)
                pos[idx, 1] += rng.uniform(-0.01, 0.01)
    return best_pos, best_radii


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def _optimize(pos, radii, deadline, seed=0):
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N + [(0.0, 0.5)] * N
    best_x = x0.copy()
    best_score = np.sum(radii)

    # Stage 1: smooth penalty, moderate weight (cheap, finds basin)
    try:
        res = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                       args=(50.0, 30.0),
                       options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-11, 'disp': False})
        if res.fun < _objective_smooth(x0, 50.0, 30.0):
            pc = res.x[:2 * N].reshape(N, 2)
            rc = np.maximum(res.x[2 * N:], 1e-6)
            pp, rp = _post_process(pc, rc)
            if _is_valid(pp, rp) and np.sum(rp) > best_score:
                best_score = np.sum(rp)
                best_x = np.concatenate([pp.flatten(), rp])
    except Exception:
        pass

    # Stage 2: smooth penalty, high weight (tighten feasibility)
    if time.time() < deadline - 12:
        try:
            res = minimize(_objective_smooth, best_x, method='L-BFGS-B', bounds=bounds,
                           args=(1000.0, 40.0),
                           options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-11, 'disp': False})
            if res.fun < _objective_smooth(best_x, 1000.0, 40.0):
                pc = res.x[:2 * N].reshape(N, 2)
                rc = np.maximum(res.x[2 * N:], 1e-6)
                pp, rp = _post_process(pc, rc)
                if _is_valid(pp, rp) and np.sum(rp) > best_score:
                    best_score = np.sum(rp)
                    best_x = np.concatenate([pp.flatten(), rp])
        except Exception:
            pass

    # Stage 3: SLSQP with STRICTLY CONVEX overlap constraint (dist^2 >= (r_i+r_j)^2)
    if time.time() < deadline - 10:
        try:
            def obj(x):
                return -np.sum(x[2 * N:])

            def c_boundary(x):
                p = x[:2 * N].reshape(N, 2)
                r = x[2 * N:]
                return np.concatenate([p[:, 0] - r, 1.0 - p[:, 0] - r, p[:, 1] - r, 1.0 - p[:, 1] - r])

            def c_overlap(x):
                p = x[:2 * N].reshape(N, 2)
                r = x[2 * N:]
                d2 = (p[_II, 0] - p[_JJ, 0]) ** 2 + (p[_II, 1] - p[_JJ, 1]) ** 2
                return d2 - (r[_II] + r[_JJ]) ** 2

            cons = [{'type': 'ineq', 'fun': c_boundary}, {'type': 'ineq', 'fun': c_overlap}]
            res = minimize(obj, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
            if res.success or (res.fun < obj(best_x)):
                pc = res.x[:2 * N].reshape(N, 2)
                rc = np.maximum(res.x[2 * N:], 1e-6)
                pp, rp = _post_process(pc, rc)
                if _is_valid(pp, rp) and np.sum(rp) > best_score:
                    best_score = np.sum(rp)
                    best_x = np.concatenate([pp.flatten(), rp])
        except Exception:
            pass

    # Stage 4: projected gradient polish (push radii to exact limits)
    if time.time() < deadline - 4:
        pc = best_x[:2 * N].reshape(N, 2)
        rc = best_x[2 * N:]
        try:
            pp, rp = _projected_polish(pc, rc, deadline, steps=20000)
            if _is_valid(pp, rp) and np.sum(rp) > best_score:
                best_score = np.sum(rp)
                best_x = np.concatenate([pp.flatten(), rp])
        except Exception:
            pass

    return best_x[:2 * N].reshape(N, 2), best_x[2 * N:]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pos = None
    best_rad = None
    best_score = 0.0

    inits = [
        _best_fit_init(seed=1, center_first=True),
        _best_fit_init(seed=2, center_first=False),
        _hex_init(),
    ]

    for i, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        pos_pp, rad_pp = _post_process(pos_init, rad_init)
        if _is_valid(pos_pp, rad_pp) and np.sum(rad_pp) > best_score:
            best_score = np.sum(rad_pp)
            best_pos, best_rad = pos_pp.copy(), rad_pp.copy()
        try:
            pos_opt, rad_opt = _optimize(pos_pp, rad_pp, deadline, seed=i)
            pos_pp, rad_pp = _post_process(pos_opt, rad_opt)
            if _is_valid(pos_pp, rad_pp) and np.sum(rad_pp) > best_score:
                best_score = np.sum(rad_pp)
                best_pos, best_rad = pos_pp.copy(), rad_pp.copy()
        except Exception:
            pass

    if best_pos is None:
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        k = 0
        for a in range(5):
            for b in range(5):
                if k < N:
                    pos[k] = [(a + 0.5) / 5.0, (b + 0.5) / 5.0]
                    rad[k] = 0.095
                    k += 1
        best_pos, best_rad = _post_process(pos, rad)
        best_score = np.sum(best_rad)

    best_pos, best_rad = _post_process(best_pos, best_rad)
    return np.column_stack([best_pos, best_rad])

# EVOLVE-BLOCK-END
