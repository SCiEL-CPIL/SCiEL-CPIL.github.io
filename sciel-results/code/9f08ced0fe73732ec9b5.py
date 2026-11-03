"""Three hundred equal circles in a circle: soft-min objective + L-BFGS-B refinement."""
import time
import numpy as np

N = 300
_DEADLINE = 120.0 - 15.0  # safety margin before the 120s kill
_rng = np.random.default_rng(123456789)


def _score(pts):
    """Raw packing radius for an (N,2) array (double precision, mirroring evaluator)."""
    p = np.asarray(pts, dtype=np.float64)
    if p.shape != (N, 2) or not np.all(np.isfinite(p)):
        return 0.0
    d = 1.0 - np.linalg.norm(p, axis=1)
    if np.any(d < 0.0):
        return 0.0
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    return min(0.5 * min_pair, d.min())


def _objective(x):
    p = x.reshape(N, 2)
    d = 1.0 - np.linalg.norm(p, axis=1)
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return -np.minimum(0.5 * dist.min(), d.min())


def _objective_grad(x):
    p = x.reshape(N, 2)
    d = 1.0 - np.linalg.norm(p, axis=1)
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = 0.5 * dist.min()
    r_cont = d.min()
    r = min(r_pair, r_cont)
    g = np.zeros_like(x)
    if r_pair <= r_cont:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        v = (p[i] - p[j]) / max(dist[i, j], 1e-12)
        g[2 * i] = 0.5 * v[0]; g[2 * i + 1] = 0.5 * v[1]
        g[2 * j] = -0.5 * v[0]; g[2 * j + 1] = -0.5 * v[1]
    else:
        k = int(np.argmin(d))
        n = np.linalg.norm(p[k])
        if n > 1e-12:
            g[2 * k] = -p[k][0] / n
            g[2 * k + 1] = -p[k][1] / n
    return g


def _hex_init():
    pts = []
    pts.append((0.0, 0.0))
    sp = 0.10
    for ring in range(1, 60):
        for k in range(6 * ring):
            a = (k / (6.0 * ring)) * 2.0 * np.pi
            r = ring * sp
            pts.append((r * np.cos(a), r * np.sin(a)))
    pts = np.array(pts)
    keep = np.linalg.norm(pts, axis=1) <= 0.995
    pts = pts[keep]
    if len(pts) < N:
        ang = np.linspace(0, 2 * np.pi, N - len(pts), endpoint=False)
        extra = np.column_stack([0.95 * np.cos(ang), 0.95 * np.sin(ang)])
        pts = np.vstack([pts, extra])
    if len(pts) > N:
        pts = pts[np.argsort(np.linalg.norm(pts, axis=1), kind='stable')[:N]]
    return pts


def _ring_init():
    pts = []
    pts.append((0.0, 0.0))
    for ring in range(1, 40):
        for k in range(6 * ring):
            a = (k / (6.0 * ring)) * 2.0 * np.pi
            r = 0.99 * ring / 39.0
            pts.append((r * np.cos(a), r * np.sin(a)))
    pts = np.array(pts)
    if len(pts) < N:
        ang = np.linspace(0, 2 * np.pi, N - len(pts), endpoint=False)
        extra = np.column_stack([0.95 * np.cos(ang), 0.95 * np.sin(ang)])
        pts = np.vstack([pts, extra])
    if len(pts) > N:
        pts = pts[np.argsort(np.linalg.norm(pts, axis=1), kind='stable')[:N]]
    return pts


def _optimize(pts, time_left):
    from scipy.optimize import minimize
    x0 = pts.reshape(-1)
    try:
        res = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B',
                       options={'maxiter': 400, 'maxfun': 400, 'gtol': 1e-6,
                                'ftol': 1e-9, 'maxls': 40,
                                'time_limit': max(1.0, time_left - 2.0)})
        if res.x is not None and np.all(np.isfinite(res.x)):
            cand = res.x.reshape(N, 2)
            if _score(cand) > 0.0:
                return cand
    except Exception:
        pass
    return pts


def construct():
    t0 = time.time()
    best = None
    best_s = -1.0

    def consider(pts):
        nonlocal best, best_s
        if pts is None or pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_s:
            best, best_s = pts.copy(), s

    consider(_hex_init())
    consider(_ring_init())

    if time.time() - t0 < _DEADLINE:
        consider(_optimize(best, _DEADLINE - (time.time() - t0)))

    # A few local restarts from the current best, if time remains.
    for scale in (0.002, 0.005, 0.01):
        if time.time() - t0 >= _DEADLINE:
            break
        left = _DEADLINE - (time.time() - t0)
        if left < 6.0:
            break
        pert = best + _rng.normal(0.0, scale, size=(N, 2))
        consider(_optimize(pert, left))

    if best is None:
        best = _hex_init()
    return best
