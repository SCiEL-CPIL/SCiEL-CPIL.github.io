"""Fresh design: smooth gradient-based circle packing in the unit square.

Maximizes the sum of 26 radii subject to disjoint interiors inside [0,1]^2.
Uses a differentiable soft-min penalty for non-overlap and boundary, gradient
descent with adaptive step, multiple diverse random restarts, a polish phase,
and a strict wall-clock deadline guard.
"""
import time
import numpy as np

N = 26

# Wall-clock budget with a safety margin (evaluator kills at 150 s).
_DEADLINE = time.time() + 142.0
_rng = np.random.default_rng(0)


def _softmax(t, beta):
    """Differentiable soft-max of a scalar/1d t: ~ max(t,0), smooth at 0."""
    t = np.asarray(t, dtype=float)
    m = np.maximum(t, 0.0)
    return (m + np.logaddexp(0.0, t / beta)) * beta


def _softmax_grad(t, beta):
    """Gradient of _softmax w.r.t. t."""
    t = np.asarray(t, dtype=float)
    return np.exp(t / beta) / (1.0 + np.exp(t / beta))


def _objective(theta, beta):
    """Return (value, grad) where value = sum(r) - lambda*(overlap+boundary)."""
    n = N
    pos = theta[:2 * n].reshape(n, 2)
    r = theta[2 * n:]
    lam = 200.0

    # --- pairwise overlap ---
    d = pos[:, None, :] - pos[None, :, :]          # (n,n,2)
    dist = np.sqrt(np.sum(d * d, axis=2) + 1e-12)  # (n,n)
    req = r[:, None] + r[None, :]                  # (n,n)
    req = np.where(np.eye(n, dtype=bool), 0.0, req)
    t = req - dist                                  # positive => overlap
    ov = np.sum(_softmax(t, beta))

    # --- boundary ---
    margin = np.minimum.reduce([r, 1.0 - r,
                                pos[:, 0], 1.0 - pos[:, 0],
                                pos[:, 1], 1.0 - pos[:, 1]])
    tb = r - margin
    bd = np.sum(_softmax(tb, beta))

    value = np.sum(r) - lam * (ov + bd)

    # --- gradients ---
    g = np.zeros_like(theta)

    # overlap gradient w.r.t. positions
    w = _softmax_grad(t, beta) / beta               # (n,n)
    w = np.where(np.eye(n, dtype=bool), 0.0, w)     # zero diagonal
    # d ov / d pos_i = - sum_j w_ij * (pos_i - pos_j)/dist_ij
    unit = d / dist[:, :, None]                     # (n,n,2)
    dpos_ov = -np.sum(w[:, :, None] * unit, axis=1)  # (n,2)
    g[:2 * n] = lam * dpos_ov.reshape(-1)
    # d ov / d r_i = sum_j w_ij
    dr_ov = np.sum(w, axis=1) + np.sum(w, axis=0)
    g[2 * n:] = lam * dr_ov

    # boundary gradient
    gb = _softmax_grad(tb, beta) / beta             # (n,)
    g[2 * n:] += lam * gb
    # boundary gradient w.r.t. positions
    # margin = min(r, 1-r, x, 1-x, y, 1-y); d margin/d x = +1 if x is the min
    x = pos[:, 0]; y = pos[:, 1]
    # which coordinate is the limiting margin for each circle
    # derivative of -margin w.r.t. x_i is -1 if x_i is the min, else 0
    is_x = (x <= y) & (x <= 1.0 - x) & (x <= 1.0 - y) & (x <= r)
    is_1mx = ((1.0 - x) <= y) & ((1.0 - x) <= x) & ((1.0 - x) <= (1.0 - y)) & ((1.0 - x) <= r)
    is_y = (y <= x) & (y <= 1.0 - y) & (y <= 1.0 - x) & (y <= r)
    is_1my = ((1.0 - y) <= x) & ((1.0 - y) <= y) & ((1.0 - y) <= (1.0 - x)) & ((1.0 - y) <= r)
    dmargin_dx = np.where(is_x, 1.0, 0.0) + np.where(is_1mx, -1.0, 0.0)
    dmargin_dy = np.where(is_y, 1.0, 0.0) + np.where(is_1my, -1.0, 0.0)
    # d bd / d x_i = -gb_i * dmargin_dx_i
    g[0::2] += -lam * gb * dmargin_dx
    g[1::2] += -lam * gb * dmargin_dy

    # sum(r) gradient
    g[2 * n:] += 1.0

    return value, g


def _feasible_clamp(theta, max_r=0.35):
    """Return a guaranteed-feasible (pos, r) from a raw theta."""
    n = N
    pos = theta[:2 * n].reshape(n, 2).copy()
    r = theta[2 * n:].copy()
    r = np.clip(r, 1e-4, max_r)
    # clamp centers so circles fit in the square
    pos[:, 0] = np.clip(pos[:, 0], r, 1.0 - r)
    pos[:, 1] = np.clip(pos[:, 1], r, 1.0 - r)
    # shrink radii until no overlap remains (conservative)
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                need = r[i] + r[j] + 1e-9
                if d < need:
                    # shrink the larger one
                    if r[i] >= r[j]:
                        r[i] = max(1e-4, (d - r[j] - 1e-9) / 2.0)
                    else:
                        r[j] = max(1e-4, (d - r[i] - 1e-9) / 2.0)
                    changed = True
        if not changed:
            break
    # final boundary clamp
    pos[:, 0] = np.clip(pos[:, 0], r, 1.0 - r)
    pos[:, 1] = np.clip(pos[:, 1], r, 1.0 - r)
    return pos, r


def _raw_value(pos, r):
    return float(np.sum(r))


def _is_valid(pos, r, tol=1e-6):
    n = len(r)
    if np.any(r < -tol) or np.any(pos < -tol) or np.any(pos > 1.0 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _optimize(theta0, time_budget, beta_start, beta_end):
    """Run gradient descent on theta0 for time_budget seconds. Returns best theta."""
    n = N
    theta = theta0.copy().astype(float)
    best_val, best_theta = -np.inf, theta.copy()
    beta = beta_start
    step = 0.002
    t0 = time.time()
    it = 0
    while time.time() - t0 < time_budget and time.time() < _DEADLINE:
        # anneal beta
        frac = min(1.0, (time.time() - t0) / max(time_budget, 1e-6))
        beta = beta_start * (beta_end / beta_start) ** frac
        val, grad = _objective(theta, beta)
        if val > best_val:
            best_val, best_theta = val, theta.copy()
        # adaptive step: scale grad to a target norm
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        g = grad / gnorm
        theta_new = theta + step * g
        val_new, _ = _objective(theta_new, beta)
        if val_new > val:
            theta = theta_new
            step = min(step * 1.05, 0.02)
        else:
            step *= 0.6
        it += 1
    return best_theta


def _make_grid_init(jitter, seed):
    rng = np.random.default_rng(seed)
    side = 6  # 6x6 grid, use 26 of 36 cells
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    idx = list(range(len(cells)))
    rng.shuffle(idx)
    chosen = [cells[k] for k in idx[:N]]
    pos = np.array(chosen)
    pos += rng.uniform(-jitter, jitter, size=pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    r = np.full(N, 0.14)
    theta = np.concatenate([pos.ravel(), r])
    return theta


def _make_random_init(seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    r = rng.uniform(0.05, 0.18, size=N)
    theta = np.concatenate([pos.ravel(), r])
    return theta


def _make_analytic_init():
    # A hand-tuned starting layout: rings + center, all inside the square.
    pos = []
    r = []
    # center
    pos.append((0.5, 0.5)); r.append(0.16)
    # inner ring of 8
    for k in range(8):
        a = 2 * np.pi * k / 8
        pos.append((0.5 + 0.30 * np.cos(a), 0.5 + 0.30 * np.sin(a)))
        r.append(0.11)
    # outer ring of 17
    for k in range(17):
        a = 2 * np.pi * k / 17 + 0.1
        pos.append((0.5 + 0.42 * np.cos(a), 0.5 + 0.42 * np.sin(a)))
        r.append(0.075)
    pos = np.array(pos[:N])
    r = np.array(r[:N])
    pos = np.clip(pos, 0.05, 0.95)
    theta = np.concatenate([pos.ravel(), r])
    return theta


def construct():
    t_start = time.time()
    best_val = -np.inf
    best_pos, best_r = None, None

    # candidate initializers (diverse)
    inits = []
    inits.append(("grid_j0", _make_grid_init(0.0, 1)))
    inits.append(("grid_j1", _make_grid_init(0.03, 7)))
    inits.append(("rand_a", _make_random_init(11)))
    inits.append(("rand_b", _make_random_init(99)))
    inits.append(("analytic", _make_analytic_init()))

    # give each init a slice of the budget
    total_budget = min(138.0, _DEADLINE - time.time())
    per_init = total_budget / len(inits)

    for name, theta0 in inits:
        if time.time() > _DEADLINE - 2.0:
            break
        # quick polish: a short run to get a feasible candidate even if we run out
        budget = min(per_init, _DEADLINE - time.time() - 2.0)
        if budget < 0.5:
            break
        theta_best = _optimize(theta0, budget, beta_start=0.05, beta_end=0.005)
        pos, r = _feasible_clamp(theta_best)
        if _is_valid(pos, r, tol=1e-5):
            v = _raw_value(pos, r)
            if v > best_val:
                best_val, best_pos, best_r = v, pos, r
        else:
            # still keep a feasible fallback
            pos, r = _feasible_clamp(theta_best)
            v = _raw_value(pos, r)
            if v > best_val:
                best_val, best_pos, best_r = v, pos, r

    # If we still have time, do a polish pass on the best found (tighter beta).
    if best_pos is not None and time.time() < _DEADLINE - 8.0:
        theta_best = np.concatenate([best_pos.ravel(), best_r])
        budget = min(40.0, _DEADLINE - time.time() - 6.0)
        if budget > 2.0:
            theta_pol = _optimize(theta_best, budget, beta_start=0.01, beta_end=0.002)
            pos, r = _feasible_clamp(theta_pol)
            if _is_valid(pos, r, tol=1e-6):
                v = _raw_value(pos, r)
                if v > best_val:
                    best_val, best_pos, best_r = v, pos, r

    # Guarantee a valid return even in the worst case.
    if best_pos is None:
        pos, r = _feasible_clamp(_make_grid_init(0.0, 1))
        best_pos, best_r = pos, r

    out = np.zeros((N, 3))
    out[:, 0] = best_pos[:, 0]
    out[:, 1] = best_pos[:, 1]
    out[:, 2] = best_r
    return out
