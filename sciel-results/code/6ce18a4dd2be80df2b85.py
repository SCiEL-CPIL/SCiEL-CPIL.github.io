import numpy as np
import time

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


def _force_grad(pos, t):
    """
    Gradient of E = (1/t) * (sum_pair exp(-t*|d_ij|) + sum_i exp(-t*(1-|p_i|))).
    Maximizing E (i.e. minimizing -E) pushes apart the closest pairs and pulls
    points inward. Returns (grad_E, E, min_pair, min_bound).
    """
    n = pos.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = pos[iu[0]] - pos[iu[1]]
    pair_d = np.linalg.norm(diff, axis=1)
    bound_d = 1.0 - np.linalg.norm(pos, axis=1)

    all_d = np.concatenate([pair_d, bound_d])
    d_min = np.min(all_d)
    shifted = all_d - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    grad = np.zeros_like(pos)

    pw = exp_terms[:len(pair_d)] / S
    dsafe = np.where(pair_d < 1e-12, 1.0, pair_d)
    unit = diff / dsafe[:, None]
    # dE/dpos_i = -w_i * (pos_i - pos_j)/d   (push apart)
    np.add.at(grad, iu[0], -pw[:, None] * unit)
    np.add.at(grad, iu[1], pw[:, None] * unit)

    bw = exp_terms[len(pair_d):] / S
    norms = np.linalg.norm(pos, axis=1)[:, None]
    norms = np.where(norms < 1e-12, 1.0, norms)
    # dE/dpos_i = -w_i * (1-|p|)/|p| * p  ->  -w_i * p/|p|  (pull inward)
    grad += -bw[:, None] * (pos / norms)

    E = (1.0 / t) * np.log(S) + d_min
    return grad, E, np.min(pair_d), np.min(bound_d)


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
    for n_rings in range(5, 17):
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


def _nesterov_optimize(points, deadline, t0=40.0, t1=4000.0,
                       lr0=0.0012, max_iter=6000, decay=0.9997,
                       mu=0.9, beta=1.0, clip=0.01):
    """
    Nesterov accelerated gradient ascent on E (force-directed energy).
    Temperature anneals from t0 (broad, global) to t1 (sharp, local).
    """
    pos = points.copy()
    vel = np.zeros_like(pos)
    best = pos.copy()
    best_s = _score(best)
    cur_t = t0
    cur_lr = lr0
    it = 0
    while it < max_iter:
        if time.time() > deadline:
            break
        g, E, _, _ = _force_grad(pos, cur_t)
        # Nesterov lookahead
        vel = mu * vel + cur_lr * g
        cand = pos + beta * vel
        # clip velocity magnitude to keep steps stable
        vnorm = np.linalg.norm(vel, axis=1)
        if np.any(vnorm > clip):
            vel *= (clip / np.where(vnorm > clip, vnorm, 1.0))[:, None]
            cand = pos + beta * vel
        # keep inside container (project)
        nrm = np.linalg.norm(cand, axis=1)
        m = nrm > 1.0
        if np.any(m):
            cand[m] = cand[m] / nrm[m, None] * 0.999
        pos = cand
        it += 1
        # anneal temperature and decay step size
        cur_t = min(cur_t * decay ** -1 if False else cur_t * (t1 / t0) ** (1.0 / max(1, max_iter)), t1)
        cur_lr = max(cur_lr * 0.9995, 1e-6)
        # periodic evaluation
        if it % 60 == 0:
            s = _score(pos)
            if s > best_s + 1e-15:
                best_s = s
                best = pos.copy()
    s = _score(pos)
    if s > best_s:
        best_s = s
        best = pos.copy()
    return best


def _refine_lbfgs(points, deadline):
    """Final tight polish using L-BFGS-B on the soft-min surrogate."""
    from scipy.optimize import minimize

    def obj_grad(x):
        p = x.reshape(N, 2)
        g, E, _, _ = _force_grad(p, 1500.0)
        return -E, (-g).flatten()

    bounds = [(-1.5, 1.5)] * (N * 2)
    try:
        res = minimize(
            obj_grad, points.flatten(), method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-11,
                     'maxls': 40, 'maxfun': 20000},
        )
        return res.x.reshape(N, 2)
    except Exception:
        return points


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY
    rng = np.random.default_rng(SEED)

    # ---- Build diverse initial configurations ----
    configs = []
    base_spacing = 2.0 / (np.sqrt(N) + 1.0)
    for angle in [0, 12, 24, 36, 48, 60]:
        sp = base_spacing
        for _ in range(30):
            pts = _lattice_seed(sp, angle)
            if pts is not None:
                configs.append(pts)
                break
            sp *= 0.98
    ring = _radial_ring_init()
    if ring is not None:
        configs.append(ring)
    for _ in range(4):
        configs.append(_random_init(rng))

    # ---- Optimize each configuration with Nesterov force-directed ascent ----
    best = None
    best_r = -np.inf
    for cfg in configs:
        if time.time() > deadline - 25.0:
            break
        cur = cfg.copy()
        cur_r = _score(cur)
        if cur_r > best_r:
            best_r, best = cur_r, cur.copy()

        # Multi-phase: broad -> sharp temperature schedules
        for (tt0, tt1, lr, mx) in [
            (30.0, 1200.0, 0.0015, 3000),
            (1200.0, 5000.0, 0.0008, 2500),
        ]:
            if time.time() > deadline - 20.0:
                break
            opt = _nesterov_optimize(cur, deadline - 18.0,
                                     t0=tt0, t1=tt1, lr0=lr, max_iter=mx)
            r = _score(opt)
            if r > cur_r:
                cur_r, cur = r, opt
            if r > best_r:
                best_r, best = r, opt.copy()

    # ---- Final tight L-BFGS polish on the best solution ----
    if best is not None and time.time() < deadline - 3.0:
        opt = _refine_lbfgs(best, deadline)
        r = _score(opt)
        if r > best_r:
            best_r, best = r, opt

    if best is None:
        best = _lattice_seed(base_spacing, 0.0)
    return best
