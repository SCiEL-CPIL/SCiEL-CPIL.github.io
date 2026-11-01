"""Packing 50 equal circles in the unit disk: force-directed settle + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # hard kill is 150s; keep a safety margin


def _score(pts):
    """Raw objective: largest common radius (double precision)."""
    pts = np.asarray(pts, dtype=float)
    d = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - d.max()
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return min(boundary, dist.min() * 0.5)


def _ring_init(n):
    """Concentric rings, 1 at centre, 10/12/14/13."""
    rings = [1, 10, 12, 14, 13]
    radii = [0.0, 0.28, 0.55, 0.78, 0.95]
    pts = [np.array([0.0, 0.0])]
    for k, r in zip(rings[1:], radii[1:]):
        ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
        pts.append(np.column_stack([r * np.cos(ang), r * np.sin(ang)]))
    return np.vstack(pts)


def _hex_init(n):
    """Dense hexagonal lattice cut to the disk, sorted by radius."""
    s = 0.205
    pts = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = i * s + 0.5 * s * (j & 1)
            y = j * s * 0.8660254037844386
            r = np.hypot(x, y)
            if r <= 1.0 - 0.125:
                pts.append((r, x, y))
    pts.sort()
    out = np.array([[x, y] for _, x, y in pts])
    if len(out) >= n:
        return out[:n]
    # pad with ring points if too few
    need = n - len(out)
    extra = _ring_init(need)
    return np.vstack([out, extra])


def _repel_forces(p, r, dt):
    """Soft repulsion for overlapping pairs + boundary push-back."""
    p = p.reshape(-1, 2)
    f = np.zeros_like(p)
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    m = dist < 2.0 * r
    if m.any():
        d2 = dist[m]
        d2 = np.clip(d2, 1e-9, None)
        dvec = d[m]
        w = np.minimum((2.0 * r - d2) / (2.0 * r), 1.0)
        f[m] -= w[:, None] * (dvec / d2[:, None])
    rad = np.sqrt((p ** 2).sum(-1))
    rad = np.clip(rad, 1e-9, None)
    over = rad > 1.0 - r
    if over.any():
        f[over] -= (p[over] / rad[over, None]) * np.clip(rad[over] - (1.0 - r), 0.0, None)
    return f * dt


def _force_settle(p0, r, dt=0.001, iters=4000, damp=0.90):
    """Velocity-Verlet-ish settle with soft repulsion and boundary forces."""
    p = p0.copy().astype(float)
    v = np.zeros_like(p)
    for _ in range(iters):
        f = _repel_forces(p, r, dt)
        v = v * damp + f
        p = p + v
    return p


def _lbfgs_obj(x, r):
    """Return -r (constant) so L-BFGS-B drives the infeasible penalty down."""
    return -r


def _lbfgs_grad(x, r):
    """Gradient of the infeasibility penalty (sum of squared violations)."""
    p = x.reshape(-1, 2)
    g = np.zeros_like(p)
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    m = dist < 2.0 * r
    if m.any():
        d2 = dist[m]
        d2 = np.clip(d2, 1e-12, None)
        dvec = d[m]
        w = (2.0 * r - d2) / d2
        g[m] += w[:, None] * dvec
    rad = np.sqrt((p ** 2).sum(-1))
    rad = np.clip(rad, 1e-12, None)
    over = rad > 1.0 - r
    if over.any():
        g[over] += (p[over] / rad[over, None]) * np.clip(rad[over] - (1.0 - r), 0.0, None)
    return g.ravel()


def _lbfgs_refine(p0, r):
    """Polish with L-BFGS-B on the infeasibility penalty."""
    try:
        x0 = p0.ravel()
        res = minimize(_lbfgs_obj, x0, jac=_lbfgs_grad, args=(r,), method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10,
                                "maxfun": 20000})
        return res.x.reshape(-1, 2)
    except Exception:
        return p0


def _refine_radius(p):
    """Locally grow r until infeasible, then polish; return best feasible."""
    best = p.copy()
    best_r = _score(best)
    r = best_r
    for _ in range(40):
        if time.time() > DEADLINE:
            break
        r2 = r * 1.002
        p2 = _lbfgs_refine(p, r2)
        s2 = _score(p2)
        if s2 >= r2 - 1e-9:
            p, r = p2, r2
            best, best_r = p2, s2
        else:
            break
    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(20240517)

    # candidate initialisations
    cands = [_ring_init(N), _hex_init(N)]
    # a few randomised ring perturbations
    for _ in range(3):
        base = _ring_init(N).copy()
        base += rng.normal(0.0, 0.01, base.shape)
        cands.append(base)

    # settle each candidate with force dynamics, then polish
    best_pts, best_r = None, -1.0
    for c in cands:
        if time.time() > DEADLINE - 20.0:
            break
        r0 = _score(c)
        settled = _force_settle(c, max(r0, 0.12))
        polished = _lbfgs_refine(settled, max(r0, 0.12))
        pts, r = _refine_radius(polished)
        if r > best_r:
            best_r, best_pts = r, pts

    # if we have time, do a final radius-growth polish on the best
    if best_pts is not None and time.time() < DEADLINE - 5.0:
        pts, r = _refine_radius(best_pts)
        if r > best_r:
            best_r, best_pts = r, pts

    # safety: guarantee a valid, finite result
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _ring_init(N)

    # final feasibility fix: clamp any point inside and nudge coincident ones
    d = np.hypot(best_pts[:, 0], best_pts[:, 1])
    over = d > 1.0
    if over.any():
        best_pts[over] = best_pts[over] / d[over, None]
    return best_pts
