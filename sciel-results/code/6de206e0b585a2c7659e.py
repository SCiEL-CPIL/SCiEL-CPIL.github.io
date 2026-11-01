"""Packing of 50 equal circles in a unit square via simulated annealing + L-BFGS polish."""
import numpy as np
import time

N = 50
HALF = 0.5


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _softmin_score(pts, t):
    """Smooth objective to maximize: soft-min of all constraints."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    vals = np.concatenate([d, pair])
    m = np.min(vals)
    s = np.sum(np.exp(-t * (vals - m)))
    return m - np.log(s) / t


def _violation_score(pts, r_target, t):
    """Smooth objective: minimize total violation of radius r_target."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0

    boundary_viol = np.maximum(0, r_target - d)
    pair_viol = np.maximum(0, r_target - pair)

    all_viol = np.concatenate([boundary_viol, pair_viol])
    if np.max(all_viol) < 1e-15:
        return 0.0
    m = np.max(all_viol)
    s = np.sum(np.exp(t * (all_viol - m)))
    return np.log(s) / t + m - 1.0 / t


def _lbfgs_maximize(pts0, t, maxiter, deadline):
    """Run L-BFGS-B to maximize the soft-min score."""
    from scipy.optimize import minimize
    x0 = pts0.ravel()

    def neg_obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_softmin_score(p, t)

    def neg_grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        eps = 1e-5
        g = np.zeros_like(x)
        base = neg_obj(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (neg_obj(xp) - base) / eps
        return g

    res = minimize(neg_obj, x0, method='L-BFGS-B',
                   jac=neg_grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(-res.fun)


def _lbfgs_violation(pts0, r_target, t, maxiter, deadline):
    """Run L-BFGS-B to minimize violation of radius r_target."""
    from scipy.optimize import minimize
    x0 = pts0.ravel()

    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_score(p, r_target, t)

    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        eps = 1e-5
        g = np.zeros_like(x)
        base = obj(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (obj(xp) - base) / eps
        return g

    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(res.fun)


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice, jittered."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF + 1e-9, HALF - 1e-9)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _simulated_annealing(pts0, rng, deadline, t_start=1.0, t_end=0.001, n_iter=None):
    """
    Simulated annealing to maximize the minimum distance (radius).
    Directly perturbs circle centers and accepts moves based on temperature.
    """
    pts = pts0.copy()
    current_r = _compute_r(pts)
    best_pts = pts.copy()
    best_r = current_r

    if n_iter is None:
        n_iter = 50000

    for it in range(n_iter):
        if time.time() > deadline:
            break

        # Temperature schedule: exponential cooling
        progress = it / n_iter
        t = t_start * (t_end / t_start) ** progress

        # Choose a random circle to perturb
        idx = rng.integers(0, N)

        # Perturbation: random step in random direction
        step_size = 0.02 * (1.0 - progress * 0.5)  # decrease step size over time
        direction = rng.normal(0, 1, size=2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        new_pts = pts.copy()
        new_pts[idx] += direction * step_size

        # Clip to bounds
        new_pts[idx] = np.clip(new_pts[idx], -HALF + 1e-9, HALF - 1e-9)

        # Compute new radius
        new_r = _compute_r(new_pts)

        # Acceptance criterion
        delta = new_r - current_r
        if delta > 0:
            # Always accept improvement
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            # Accept with probability exp(delta / t)
            if t > 1e-15:
                prob = np.exp(delta / t)
                if rng.random() < prob:
                    pts = new_pts
                    current_r = new_r
            # Otherwise reject (keep current)

    return best_pts, best_r


def _sa_with_restarts(pts0, rng, deadline, n_restarts=3):
    """
    Run simulated annealing with multiple restarts from the initial point.
    Each restart uses a different random perturbation of the starting point.
    """
    best_pts = pts0.copy()
    best_r = _compute_r(pts0)

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        # Perturb the starting point for this restart
        jitter = 0.01 * (restart + 1)
        pts_perturbed = pts0 + rng.normal(0, jitter, size=pts0.shape)
        pts_perturbed = np.clip(pts_perturbed, -HALF + 1e-9, HALF - 1e-9)

        # Run SA
        sa_pts, sa_r = _simulated_annealing(
            pts_perturbed, rng, deadline,
            t_start=0.5, t_end=0.0005,
            n_iter=30000
        )

        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()

    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s
    rng = np.random.default_rng(12345)

    best_pts = None
    best_r = 0.0

    def try_update(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 1: Multi-start with L-BFGS to get a good initial solution
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
    ]

    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        try:
            pts, score = _lbfgs_maximize(init, t=20.0, maxiter=150, deadline=deadline)
            try_update(pts)
        except Exception:
            pass

    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Phase 2: Simulated annealing with restarts to explore the landscape
    if time.time() < deadline - 40.0:
        sa_pts, sa_r = _sa_with_restarts(best_pts, rng, deadline, n_restarts=3)
        try_update(sa_pts)

    # Phase 3: Polish with L-BFGS using sharper soft-min
    if time.time() < deadline - 15.0:
        for t in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                pts, score = _lbfgs_maximize(best_pts, t=t, maxiter=200, deadline=deadline)
                try_update(pts)
            except Exception:
                pass

    # Phase 4: Binary search on radius using violation minimization
    if time.time() < deadline - 15.0:
        r_low = best_r
        r_high = min(r_low + 0.01, 0.1)

        for iteration in range(12):
            if time.time() > deadline - 10.0:
                break
            r_mid = (r_low + r_high) / 2.0

            try:
                pts, viol = _lbfgs_violation(best_pts, r_mid, t=50.0, maxiter=200, deadline=deadline)
                try_update(pts)
            except Exception:
                pass

            r_achieved = _compute_r(best_pts)
            if r_achieved >= r_mid - 1e-8:
                r_low = r_mid
            else:
                r_high = r_mid

    # Phase 5: Final polish
    if time.time() < deadline - 5.0:
        for t in [150.0, 300.0]:
            if time.time() > deadline - 3.0:
                break
            try:
                pts, score = _lbfgs_maximize(best_pts, t=t, maxiter=300, deadline=deadline)
                try_update(pts)
            except Exception:
                pass

    # Phase 6: One more SA pass from the best found
    if time.time() < deadline - 5.0:
        try:
            sa_pts, sa_r = _simulated_annealing(
                best_pts, rng, deadline,
                t_start=0.2, t_end=0.0001,
                n_iter=20000
            )
            try_update(sa_pts)
        except Exception:
            pass

    return best_pts
