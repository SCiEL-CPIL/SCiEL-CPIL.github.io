"""Circle packing in perimeter-4 rectangle.

Combines:
  * Current program: penalty/L-BFGS-B optimizer + local refinement + clamped evaluation.
  * Inspiration: force-based radius-growth relaxation and space-filling (Hilbert/Z-order)
    curve initializations.
  * Structural change: the rectangle aspect ratio (W, with H=2-W) is a search variable
    explored via a coarse-to-fine scan, rather than being fixed per candidate.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 138.0  # safety margin before 150s kill


def _time_ok():
    return time.time() < DEADLINE


# ---------------------------------------------------------------------------
# Validity / evaluation
# ---------------------------------------------------------------------------
def _is_valid(circles, W, H, tol=1e-7):
    """Strict feasibility check: inside rectangle, no overlaps."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 1e-9:
            return False
        if x < r - tol or x > W - r + tol or y < r - tol or y > H - r + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            if dx * dx + dy * dy < (circles[i][2] + circles[j][2]) ** 2 - tol:
                return False
    return True


def _clamp_and_sum(circles, W, H):
    """Clamp a candidate to feasibility; return (sum_r, valid_bool)."""
    n = len(circles)
    cx = np.array([c[0] for c in circles], dtype=np.float64)
    cy = np.array([c[1] for c in circles], dtype=np.float64)
    r = np.array([c[2] for c in circles], dtype=np.float64)
    r = np.maximum(r, 1e-7)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-7)
    # resolve residual overlaps by shrinking smaller circle
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            req = r[i] + r[j]
            if req > d + 1e-9:
                # shrink the smaller one to fit
                if r[i] <= r[j]:
                    r[i] = max(1e-7, (d - r[j]))
                else:
                    r[j] = max(1e-7, (d - r[i]))
    # re-clamp to walls after shrink (radii only decreased, so fine)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-7)
    out = np.column_stack([cx, cy, r])
    valid = _is_valid(out, W, H)
    return float(np.sum(r)), valid


# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------
def _hilbert_xy(n, order=8):
    size = 1 << order

    def d2xy(h, size):
        x = y = 0
        for s in range(1, size, 2):
            rx = 1 & (h // 2)
            ry = 1 & (h ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
        return x, y

    pts = []
    for i in range(n):
        h = int(i * (size - 1) / max(n - 1, 1))
        x, y = d2xy(h, size)
        pts.append(((x + 0.5) / size, (y + 0.5) / size))
    return pts


def _zorder_xy(n, order=8):
    size = 1 << order

    def z2xy(z, size):
        x = y = 0
        for s in range(size):
            x |= ((z >> (2 * s)) & 1) << s
            y |= ((z >> (2 * s + 1)) & 1) << s
        return x, y

    pts = []
    for i in range(n):
        z = int(i * (size * size - 1) / max(n - 1, 1))
        x, y = z2xy(z, size)
        pts.append(((x + 0.5) / size, (y + 0.5) / size))
    return pts


def _grid_xy(kx, ky):
    pts = []
    for j in range(ky):
        for i in range(kx):
            pts.append(((i + 0.5) / kx, (j + 0.5) / ky))
    return pts[:N]


def _make_init(kind, W, H, base_r):
    if kind == "hilbert":
        pts = _hilbert_xy(N)
    elif kind == "zorder":
        pts = _zorder_xy(N)
    else:
        # grid: pick kx*ky >= N closest to aspect
        kx = max(1, int(round(np.sqrt(N * W / H))))
        ky = max(1, int(round(N / kx)))
        if kx * ky < N:
            ky += 1
        pts = _grid_xy(kx, ky)
    circles = [(u * W, v * H, base_r) for (u, v) in pts]
    return circles


# ---------------------------------------------------------------------------
# Force-based relaxation with radius growth
# ---------------------------------------------------------------------------
def _relax(circles, W, H, max_iter=4000, growth_rate=0.0006, lr=0.12,
           time_limit=None, seed=None):
    if seed is not None:
        random.seed(seed)
    xs = np.array([c[0] for c in circles], dtype=np.float64)
    ys = np.array([c[1] for c in circles], dtype=np.float64)
    rs = np.array([c[2] for c in circles], dtype=np.float64)
    n = len(xs)
    t0 = time.time()
    for _ in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        rs_new = rs * (1.0 + growth_rate)
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                min_dist = rs_new[i] + rs_new[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-8:
                        nx, ny = dx / dist, dy / dist
                        fx[i] += nx * overlap
                        fy[i] += ny * overlap
                        fx[j] -= nx * overlap
                        fy[j] -= ny * overlap
                    else:
                        ang = random.uniform(0, 2 * np.pi)
                        fx[i] += np.cos(ang) * overlap
                        fy[i] += np.sin(ang) * overlap
                        fx[j] -= np.cos(ang) * overlap
                        fy[j] -= np.sin(ang) * overlap
        for i in range(n):
            if xs[i] < rs_new[i]:
                fx[i] += (rs_new[i] - xs[i]) * 5.0
            if xs[i] > W - rs_new[i]:
                fx[i] -= (xs[i] - (W - rs_new[i])) * 5.0
            if ys[i] < rs_new[i]:
                fy[i] += (rs_new[i] - ys[i]) * 5.0
            if ys[i] > H - rs_new[i]:
                fy[i] -= (ys[i] - (H - rs_new[i])) * 5.0
        xs_new = xs + lr * fx
        ys_new = ys + lr * fy
        xs_new = np.clip(xs_new, rs_new, W - rs_new)
        ys_new = np.clip(ys_new, rs_new, H - rs_new)
        # validity check
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs_new[i] - xs_new[j]
                dy = ys_new[i] - ys_new[j]
                if dx * dx + dy * dy < (rs_new[i] + rs_new[j]) ** 2 - 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            xs, ys, rs = xs_new, ys_new, rs_new
        else:
            growth_rate *= 0.93
            if growth_rate < 1e-7:
                break
    return list(zip(xs, ys, rs))


# ---------------------------------------------------------------------------
# Penalty optimizer (current program's machinery)
# ---------------------------------------------------------------------------
def _penalty(x, W, H):
    n = (len(x) - 2) // 3
    x = np.array(x[:2 + 3 * n]).reshape(-1, 3)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    obj = -np.sum(r)
    pen = 0.0
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    pen += 1e4 * np.sum(np.maximum(r - lo, 0.0) ** 2)
    pen += 1e4 * np.sum(np.maximum(-r, 0.0) ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            req = r[i] + r[j]
            if req - d > 0:
                pen += 1e4 * (req - d) ** 2
    return obj + pen


def _optimize(circles, W, H, time_limit=None):
    x0 = np.concatenate([[W, H]] + np.array(circles).flatten())
    m = (len(x0) - 2) // 3

    def fun(x):
        return _penalty(x, W, H)

    def grad(x):
        h = 1e-5
        g = np.zeros(len(x))
        f0 = fun(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (fun(xp) - fun(xm)) / (2 * h)
        return g

    best = circles
    best_sum, _ = _clamp_and_sum(circles, W, H)
    try:
        res = minimize(fun, x0, method="L-BFGS-B", jac=grad,
                       options={"maxiter": 1500, "maxls": 20, "gtol": 1e-6, "iprint": 0})
        if res.x is not None and time.time() < (time_limit or DEADLINE):
            cand = [tuple(res.x[2 + 3 * k:5 + 3 * k]) for k in range(m)]
            s, v = _clamp_and_sum(cand, W, H)
            if s > best_sum:
                best, best_sum = cand, s
    except Exception:
        pass
    return best, best_sum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    best_sum = 0.0
    best_arr = None
    rng = random.Random(12345)

    # Coarse-to-fine scan over aspect ratio W (H = 2 - W).
    # Symmetric around 1.0; finer grid near the optimum.
    W_grid = np.round(np.concatenate([
        np.arange(0.55, 1.45, 0.05),
        np.arange(0.95, 1.06, 0.01),
    ]), 4)
    W_grid = [float(w) for w in W_grid if 0.3 < w < 1.7]

    inits = ["hilbert", "zorder", "grid"]
    base_r = 0.06

    # Pre-build the set of (W, H, init_circles) to try, prioritized by a quick
    # relaxation score so we spend budget on promising aspect ratios.
    # First pass: cheap relaxation to rank aspect ratios.
    ranked = []
    for W in W_grid:
        H = 2.0 - W
        for kind in inits:
            circ = _make_init(kind, W, H, base_r)
            circ = _relax(circ, W, H, max_iter=600, growth_rate=0.0008,
                          lr=0.1, time_limit=DEADLINE - 5, seed=rng.randint(0, 10**6))
            s, v = _clamp_and_sum(circ, W, H)
            ranked.append((s, W, H, kind, circ))
        if not _time_ok():
            break
    ranked.sort(key=lambda t: -t[0])

    # Second pass: refine the top candidates with full relax + optimizer.
    for (s0, W, H, kind, circ) in ranked[:12]:
        if not _time_ok():
            break
        # full relaxation
        circ = _relax(circ, W, H, max_iter=4000, growth_rate=0.0005,
                      lr=0.12, time_limit=DEADLINE - 8, seed=rng.randint(0, 10**6))
        s, _ = _clamp_and_sum(circ, W, H)
        # optimizer refinement
        circ, s = _optimize(circ, W, H, time_limit=DEADLINE - 6)
        if s > best_sum:
            best_sum = s
            best_arr = (W, H, circ)

    if best_arr is None:
        # fallback: plain square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    W, H, circ = best_arr
    arr, valid = _clamp_and_sum(circ, W, H)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return arr
