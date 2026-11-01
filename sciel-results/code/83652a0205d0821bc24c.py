import numpy as np
import time
from scipy.optimize import minimize

N = 300
SEED = 12345
DEADLINE = 150.0
SAFETY = 15.0


def _score(points):
    """Feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    iu = np.triu_indices(N, k=1)
    d = np.linalg.norm(points[iu[0]] - points[iu[1]], axis=1)
    r_pair = 0.5 * np.min(d)
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _soft_min_grad(pos, t):
    """
    Gradient (ascent) of the log-sum-exp soft-min of all pairwise and boundary
    distances. Returns (grad, soft_min_value, actual_min).
    """
    n = pos.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = pos[iu[0]] - pos[iu[1]]
    pair_d = np.linalg.norm(diff, axis=1)
    boundary_d = 1.0 - np.linalg.norm(pos, axis=1)

    all_d = np.concatenate([pair_d, boundary_d])
    d_min = np.min(all_d)
    shifted = all_d - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    grad = np.zeros_like(pos)

    # Pairwise: d(pair_d)/d(pos_i) = (pos_i - pos_j)/d, d/d(pos_j) = (pos_j - pos_i)/d
    pw = exp_terms[:len(pair_d)] / S
    dsafe = np.where(pair_d < 1e-12, 1e-12, pair_d)
    unit = diff / dsafe[:, None]
    np.add.at(grad, iu[0], pw[:, None] * unit)
    np.add.at(grad, iu[1], -pw[:, None] * unit)

    # Boundary: d(1 - |p|)/d(p) = -p/|p|
    bw = exp_terms[len(pair_d):] / S
    norms = np.linalg.norm(pos, axis=1)[:, None]
    norms = np.where(norms < 1e-12, 1e-12, norms)
    grad += bw[:, None] * (-pos / norms)

    soft_min_val = (1.0 / t) * np.log(S) + d_min
    return grad, soft_min_val, d_min


def _lattice_seed(spacing, angle_deg=0.0):
    """Hexagonal lattice points inside the unit disk, optionally rotated."""
    angle = np.radians(angle_deg)
    ca, sa = np.cos(angle), np.sin(angle)
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            pts.append((x * ca - y * sa, x * sa + y * ca))
    pts = np.array(pts)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[rad <= 1.0 - spacing / 2.0]
    if len(pts) < N:
        return None
    order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
    return pts[order[:N]].copy()


def _radial_ring_init():
    """Best concentric-ring configuration over a range of ring counts."""
    best = None
    best_s = -np.inf
    for n_rings in range(5, 16):
        weights = [2 * np.pi * ((i + 1) / n_rings * 0.95) for i in range(n_rings)]
        total_w = sum(weights)
        counts = []
        remaining = N
        for i in range(n_rings - 1):
            c = max(1, int(round(N * weights[i] / total_w)))
            c = min(c, remaining - (n_rings - 1 - i))
            counts.append(c)
            remaining -= c
        counts.append(remaining)
        pts = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r_i = (i + 1) / n_rings * 0.95
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                pts.append((r_i * np.cos(a), r_i * np.sin(a)))
        if len(pts) != N:
            continue
        pts = np.array(pts)
        s = _score(pts)
        if s > best_s:
            best_s = s
            best = pts.copy()
    return best


def _random_init(rng):
    pts = rng.standard_normal((N, 2))
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, None] * 0.95
    return pts


def _optimize_lbfgs(points, beta, maxiter, deadline):
    """L-BFGS-B on the soft-min surrogate (maximize min distance)."""
    def obj_grad(x):
        p = x.reshape(N, 2)
        g, val, _ = _soft_min_grad(p, beta)
        return -val, (-g).flatten()

    bounds = [(-1.5, 1.5)] * (N * 2)
    res = minimize(
        obj_grad, points.flatten(), method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10,
                 'maxls': 40, 'maxfun': 20000},
    )
    return res.x.reshape(N, 2)


def _gd_polish(points, deadline, rng, t=200.0, max_iter=400, lr=0.004):
    """Gradient-ascent polish on the soft-min surrogate."""
    best = points.copy()
    best_s = _score(best)
    cur_lr = lr
    patience = 0
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        g, _, _ = _soft_min_grad(best, t)
        new = best + cur_lr * g
        norms = np.linalg.norm(new, axis=1)
        m = norms > 1.0
        if np.any(m):
            new[m] = new[m] / norms[m, None] * 0.999
        s = _score(new)
        if s > best_s + 1e-15:
            best = new
            best_s = s
            patience = 0
            cur_lr = min(cur_lr * 1.01, 0.012)
        else:
            patience += 1
            cur_lr *= 0.95
        if patience >= 40:
            break
    return best


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY
    rng = np.random.default_rng(SEED)

    # ---- Build diverse initial configurations ----
    configs = []
    base_spacing = 2.0 / (np.sqrt(N) + 1.0)
    for angle in [0, 15, 30, 45, 60]:
        sp = base_spacing
        for _ in range(25):
            pts = _lattice_seed(sp, angle)
            if pts is not None:
                configs.append(pts)
                break
            sp *= 0.98
    ring = _radial_ring_init()
    if ring is not None:
        configs.append(ring)
    for _ in range(3):
        configs.append(_random_init(rng))

    # ---- Optimize each configuration with a temperature schedule ----
    best = None
    best_r = -np.inf
    for cfg in configs:
        if time.time() > deadline - 20.0:
            break
        cur = cfg.copy()
        cur_r = _score(cur)
        if cur_r > best_r:
            best_r, best = cur_r, cur.copy()

        for beta in [10.0, 50.0, 150.0, 400.0, 800.0]:
            if time.time() > deadline - 15.0:
                break
            try:
                opt = _optimize_lbfgs(cur, beta, maxiter=150, deadline=deadline)
            except Exception:
                continue
            r = _score(opt)
            if r > cur_r:
                cur_r, cur = r, opt
            if r > best_r:
                best_r, best = r, opt.copy()

    # ---- High-temperature gradient-ascent polish on the best solution ----
    if best is not None and time.time() < deadline - 2.0:
        try:
            polished = _gd_polish(best, deadline, rng, t=200.0, max_iter=400, lr=0.004)
            r = _score(polished)
            if r > best_r:
                best_r, best = r, polished
        except Exception:
            pass

    # ---- Final tight L-BFGS polish ----
    if best is not None and time.time() < deadline - 2.0:
        try:
            opt = _optimize_lbfgs(best, beta=1500.0, maxiter=120, deadline=deadline)
            r = _score(opt)
            if r > best_r:
                best_r, best = r, opt
        except Exception:
            pass

    if best is None:
        best = _lattice_seed(base_spacing, 0.0)
    return best
