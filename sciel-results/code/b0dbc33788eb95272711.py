"""Hexagon packing: 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, a):
    """Vertices of a regular hexagon (side s) centered (cx,cy), rotation a rad."""
    ang = np.arange(6) * (math.pi / 3.0) + a
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)

def _support(poly, d):
    """Max of dot(poly, d) over polygon vertices."""
    return float(np.max(poly @ d))

def _max_separation(inner, outer):
    """Largest min-separation over all vertex-vertex pairs (inner vs outer)."""
    best = 0.0
    n_in = len(inner)
    n_out = len(outer)
    for i in range(n_in):
        for j in range(n_out):
            pi = inner[i]
            po = outer[j]
            # all pairs
            d = pi[:, None, :] - po[None, :, :]
            dist = np.sqrt((d * d).sum(-1))
            best = max(best, float(dist.min()))
    return best

def _min_outer_side(inner, cx, cy, rot, side, max_iter=14):
    """Minimize outer side length (center/rotation fixed) so inner hexes fit."""
    def sfun(s):
        outer = _hex_verts(cx, cy, s, rot)
        return _max_separation(inner, outer)
    lo, hi = 1.0, 40.0
    # find a hi with sfun(hi) <= 0 (inner fully inside)
    while sfun(hi) > 0.0 and hi < 200.0:
        hi *= 1.5
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if sfun(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi

def _compute_min_outer_side(inner, cx, cy, rot):
    """Compute minimal outer side length for fixed inner + center + rotation."""
    return _min_outer_side(inner, cx, cy, rot, 1.0)

def _cluster_initial():
    """Hexagonal cluster 3-4-4-3 of unit hexagons."""
    dx = math.sqrt(3.0)          # horizontal spacing between touching hexes
    dy = 1.5                      # vertical spacing between rows
    rows = [3, 4, 4, 3]
    centers = []
    for r, cnt in enumerate(rows):
        y = -1.5 + r * dy
        x0 = -0.5 * (cnt - 1) * dx
        for k in range(cnt):
            centers.append((x0 + k * dx, y))
    return centers

def _inner_polys(centers):
    polys = []
    for (x, y) in centers:
        polys.append(_hex_verts(x, y, 1.0, 0.0))
    return polys

def _objective(p, cx, cy, rot):
    """Return (outer_side, max_sep) for a candidate placement."""
    centers = [(p[2 * k], p[2 * k + 1]) for k in range(N)]
    inner = _inner_polys(centers)
    s = _min_outer_side(inner, cx, cy, rot, 1.0)
    outer = _hex_verts(cx, cy, s, rot)
    sep = _max_separation(inner, outer)
    return s, sep

def construct():
    t0 = time.time()
    deadline = t0 + 115.0   # safety margin under 120s

    # Start from the tight hexagonal cluster, centered at origin.
    centers0 = _cluster_initial()
    cx, cy, rot = 0.0, 0.0, 0.0
    p0 = np.array([v for c in centers0 for v in c], dtype=float)

    # Baseline
    s_best, sep_best = _objective(p0, cx, cy, rot)
    p_best = p0.copy()
    s_best = min(s_best, 40.0)

    # Local refinement: minimize outer side length with a smooth-ish penalty
    # via the max-separation (nonsmooth) but using gradient-free L-BFGS-B
    # on a smooth surrogate: side + lambda * max(0, -sep) is handled by
    # binary search; instead we directly minimize the computed side length.
    def obj(p):
        s, sep = _objective(p, cx, cy, rot)
        return s
    def obj_grad(p):
        # numerical gradient (central differences)
        eps = 1e-5
        g = np.zeros_like(p)
        base = obj(p)
        for i in range(len(p)):
            pp = p.copy(); pp[i] += eps
            pm = p.copy(); pm[i] -= eps
            g[i] = (obj(pp) - obj(pm)) / (2 * eps)
        return g

    # Try a few random restarts / perturbations of the cluster.
    rng = np.random.default_rng(0)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Perturb the cluster: small random offsets + slight rotation of whole set
        theta = rng.uniform(-0.15, 0.15)
        ct, st = math.cos(theta), math.sin(theta)
        p = p0.copy()
        for k in range(N):
            x, y = p[2 * k], p[2 * k + 1]
            xr = ct * x - st * y + rng.normal(0, 0.05)
            yr = st * x + ct * y + rng.normal(0, 0.05)
            p[2 * k] = xr
            p[2 * k + 1] = yr

        # Local polish with L-BFGS-B on the smooth objective (side length)
        res = None
        try:
            res = minimize(obj, p, method='L-BFGS-B',
                           options={'maxiter': 60, 'maxfun': 800,
                                    'maxls': 20, 'gtol': 1e-6})
        except Exception:
            res = None
        if res is not None and res.fun < s_best:
            s_best = res.fun
            p_best = res.x.copy()

        # Also keep best raw evaluation if optimizer underperforms
        s_now, _ = _objective(p, cx, cy, rot)
        if s_now < s_best:
            s_best = s_now
            p_best = p.copy()

        # Early stop if we've converged tightly and have budget to spare
        if s_best < 4.0 and attempts >= 3:
            break

    # Final high-precision polish on the best placement
    try:
        res = minimize(obj, p_best, method='L-BFGS-B',
                       options={'maxiter': 120, 'maxfun': 2000,
                                'maxls': 30, 'gtol': 1e-9})
        if res.fun < s_best:
            s_best = res.fun
            p_best = res.x.copy()
    except Exception:
        pass

    # Build the inner hexagon rows: (x, y, angle_degrees)
    inner = [[float(p_best[2 * k]), float(p_best[2 * k + 1]), 0.0] for k in range(N)]
    outer_side = float(min(s_best, 40.0))
    return inner, (cx, cy), outer_side, rot

# EVOLVE-BLOCK-END
