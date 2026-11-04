"""Circle packing in perimeter-4 rectangle: force-directed contact-graph optimizer."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 120.0
_START = time.time()

def _remaining():
    return DEADLINE - (time.time() - _START)

def _make_initial(w, h, seed):
    rng = np.random.default_rng(seed)
    # start from a grid jittered, then scale to fit inside [0,w]x[0,h]
    side = int(np.ceil(np.sqrt(N)))
    pts = np.zeros((N, 2))
    for i in range(N):
        gi = i % side
        gj = i // side
        pts[i, 0] = (gi + 0.5) / side
        pts[i, 1] = (gj + 0.5) / side
    # jitter
    pts += rng.uniform(-0.3, 0.3, size=(N, 2))
    # scale into rectangle with small margin
    pts[:, 0] = pts[:, 0] * w
    pts[:, 1] = pts[:, 1] * h
    r = np.full(N, 0.5 / side)
    return pts, r

def _soft_min(d, lam):
    # soft min of distances; returns a smooth lower-bound-ish surrogate
    m = np.min(d)
    return m - np.log(np.sum(np.exp((m - d) / lam))) * lam

def _evaluate(pts, r, w, h, lam):
    # objective: maximize sum(r) subject to:
    #   pair separation: r_i + r_j <= dist_ij
    #   wall: r_i <= min(x_i, w - x_i, y_i, h - y_i)
    n = len(r)
    # pair distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
    # violations (positive = infeasible)
    viol = 0.0
    # pair constraints
    for i in range(n):
        for j in range(i + 1, n):
            gap = d[i, j] - (r[i] + r[j])
            if gap < 0:
                viol += gap * gap
    # wall constraints
    for i in range(n):
        m = min(pts[i, 0], w - pts[i, 0], pts[i, 1], h - pts[i, 1])
        gap = m - r[i]
        if gap < 0:
            viol += gap * gap
    return np.sum(r), viol, d

def _force_step(pts, r, w, h, lr, iters, repulse, attract):
    n = len(r)
    for _ in range(iters):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # repulsion between overlapping pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.hypot(dx, dy)
                if dist < 1e-9:
                    dx = 1e-6
                    dist = 1e-6
                need = r[i] + r[j]
                if dist < need:
                    f = repulse * (need - dist) / dist
                    fx[i] += f * dx
                    fy[i] += f * dy
                    fx[j] -= f * dx
                    fy[j] -= f * dy
        # attraction: pull centers toward each other to tighten packing
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.hypot(dx, dy)
                if dist < 1e-9:
                    continue
                f = attract / dist
                fx[i] -= f * dx
                fy[i] -= f * dy
                fx[j] += f * dx
                fy[j] += f * dy
        # wall forces (push back inside)
        for i in range(n):
            if pts[i, 0] < r[i]:
                fx[i] += repulse * (r[i] - pts[i, 0])
            if pts[i, 0] > w - r[i]:
                fx[i] -= repulse * (pts[i, 0] - (w - r[i]))
            if pts[i, 1] < r[i]:
                fy[i] += repulse * (r[i] - pts[i, 1])
            if pts[i, 1] > h - r[i]:
                fy[i] -= repulse * (pts[i, 1] - (h - r[i]))
        # update
        pts += lr * np.stack([fx, fy], axis=1)
        # clamp inside walls
        for i in range(n):
            pts[i, 0] = min(max(pts[i, 0], r[i]), w - r[i])
            pts[i, 1] = min(max(pts[i, 1], r[i]), h - r[i])
    return pts

def _polish(pts, r, w, h):
    # high-precision local solve: maximize sum(r) with smooth constraints
    n = len(r)
    def obj(vars):
        p = vars[:2 * n].reshape(n, 2)
        rad = vars[2 * n:]
        # soft constraint penalty
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                gap = dist - (rad[i] + rad[j])
                if gap < 0:
                    pen += gap * gap
        for i in range(n):
            m = min(p[i, 0], w - p[i, 0], p[i, 1], h - p[i, 1])
            gap = m - rad[i]
            if gap < 0:
                pen += gap * gap
        return -np.sum(rad) + 1e4 * pen
    x0 = np.concatenate([pts.ravel(), r])
    res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
    if res.success or res.fun < obj(x0):
        p = res.x[:2 * n].reshape(n, 2)
        rad = res.x[2 * n:]
        # clamp radii non-negative and inside walls
        rad = np.maximum(rad, 1e-6)
        for i in range(n):
            m = min(p[i, 0], w - p[i, 0], p[i, 1], h - p[i, 1])
            rad[i] = min(rad[i], m)
        return p, rad
    return pts, r

def _improve_radii(pts, w, h):
    # grow radii as large as possible given fixed centers
    n = len(pts)
    # compute max radius per center from walls
    wall_r = np.array([min(pts[i, 0], w - pts[i, 0], pts[i, 1], h - pts[i, 1]) for i in range(n)])
    # iterative: set each r_i to min(wall_r_i, min over j of (dist_ij - r_j))
    r = np.full(n, 1e-6)
    for _ in range(200):
        changed = False
        for i in range(n):
            maxr = wall_r[i]
            for j in range(n):
                if i == j:
                    continue
                dist = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                maxr = min(maxr, dist - r[j])
            maxr = max(maxr, 1e-6)
            if maxr > r[i] + 1e-9:
                r[i] = maxr
                changed = True
        if not changed:
            break
    return r

def _feasible(pts, r, w, h, tol=1e-6):
    n = len(r)
    for i in range(n):
        if r[i] < 0:
            return False
        m = min(pts[i, 0], w - pts[i, 0], pts[i, 1], h - pts[i, 1])
        if r[i] > m + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _make_result(pts, r, w, h):
    # return array (n,3): x, y, r
    out = np.zeros((N, 3))
    for i in range(N):
        out[i, 0] = pts[i, 0]
        out[i, 1] = pts[i, 1]
        out[i, 2] = r[i]
    return out

def construct():
    best = None
    best_val = -1.0
    best_result = None

    # try several aspect ratios; perimeter fixed at 4 => w + h = 2
    aspect_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0, 2.5, 3.0]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    for aspect in aspect_list:
        if _remaining() < 12.0:
            break
        w = aspect / (1 + aspect) * 2.0
        h = 2.0 - w
        for seed in seeds:
            if _remaining() < 10.0:
                break
            pts, r = _make_initial(w, h, seed)
            # force-directed relaxation with growing radii
            it = 0
            while _remaining() > 10.0 and it < 40:
                # grow radii to max given current centers
                r = _improve_radii(pts, w, h)
                # force step to relieve overlaps
                pts = _force_step(pts, r, w, h, lr=0.02, iters=200, repulse=5.0, attract=0.5)
                it += 1
            # polish
            if _remaining() > 8.0:
                pts, r = _polish(pts, r, w, h)
            # final radii grow
            r = _improve_radii(pts, w, h)
            # validate
            if _feasible(pts, r, w, h, tol=1e-5):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = _make_result(pts, r, w, h)
    if best_result is None:
        # fallback to seed grid in a 1x1 square (perimeter 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best_result
