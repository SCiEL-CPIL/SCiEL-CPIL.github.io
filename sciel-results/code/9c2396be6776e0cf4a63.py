"""Fresh design: direct gradient-ascent optimization of the diameter-2
product-of-distances objective for 8 planar points, with random restarts
and a strict wall-clock deadline guard."""
import math
import time
import random
import numpy as np

# ---------- fixed problem constants ----------
N = 8
NP = N * (N - 1) // 2            # 28 unordered pairs
_IDX = [(i, j) for i in range(N) for j in range(i + 1, N)]
# gradient of the log-product term: sum over j!=i of (p_i - p_j)/d_ij  -> shape (N,2)
_GRAD_W = np.zeros((N, N))
for (i, j) in _IDX:
    _GRAD_W[i, j] = 1.0
    _GRAD_W[j, i] = 1.0

# ---------- objective (exact, for scoring/best-tracking) ----------
def _logprod(pts):
    d = np.sqrt(np.maximum(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2), 1e-300))
    np.fill_diagonal(d, 1.0)
    return 2.0 * np.sum(np.log(d))          # log of product over ordered pairs


def _score(pts):
    d = np.sqrt(np.maximum(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2), 1e-300))
    np.fill_diagonal(d, 1.0)
    dmax = d.max()
    if dmax <= 0.0:
        return -1e18
    s = 2.0 / dmax
    return 2.0 * (np.sum(np.log(d)) + NP * math.log(s))


# ---------- smooth objective + gradient (for ascent) ----------
def _obj_grad(pts, c):
    d = np.sqrt(np.maximum(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2), 1e-300))
    np.fill_diagonal(d, 1.0)
    # logsumexp over off-diagonal entries (diagonal is 0 -> log 0 = -inf, excluded)
    off = d[np.triu_indices(N, k=1)]
    m = off.max()
    lse = m + math.log(np.exp(c * off - c * m).sum())   # = logsumexp(c*off)
    dmax_s = lse / c
    val = 2.0 * (np.sum(np.log(d)) + NP * (math.log(2.0) - math.log(dmax_s)))
    # gradient of sum log d_ij
    G = np.zeros((N, 2))
    for (i, j) in _IDX:
        v = pts[i] - pts[j]
        dij = d[i, j]
        G[i] += v / dij
        G[j] -= v / dij
    # gradient of -NP*log(dmax_s); dmax_s = (1/c) logsumexp(c d)
    w = np.exp(c * off - lse)                 # softmax weights over pairs
    Gp = np.zeros((N, 2))
    for k, (i, j) in enumerate(_IDX):
        v = pts[i] - pts[j]
        Gp[i] += w[k] * v / d[i, j]
        Gp[j] -= w[k] * v / d[i, j]
    grad = 2.0 * (G - (NP / dmax_s) * Gp)
    return val, grad


def _reg_octagon(r):
    a = 2.0 * math.pi * np.arange(N) / N
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _regular_octagon():
    a = 2.0 * math.pi * np.arange(N) / N
    return np.column_stack([np.cos(a), np.sin(a)])


def construct():
    t0 = time.time()
    DEADLINE = 140.0          # safety margin under the 150 s kill
    rng = random.Random(12345)

    best_pts = _regular_octagon().copy()
    best_val = _score(best_pts)

    # candidate initializers
    starts = [_regular_octagon()]
    for _ in range(4):
        starts.append(_reg_octagon(rng.uniform(0.6, 1.4)))
    starts.append(np.column_stack([
        np.cos(2 * math.pi * np.arange(N) / N),
        np.sin(2 * math.pi * np.arange(N) / N),
    ]) * np.array([1.0, 1.0]))
    # a few random jittered starts
    for _ in range(3):
        base = _regular_octagon()
        base = base + rng.gauss(0, 0.25, base.shape)
        starts.append(base)

    c = 6.0                   # soft-max sharpness
    step = 0.05

    for s in range(len(starts)):
        if time.time() > DEADLINE:
            break
        pts = starts[s].copy()
        cur = _obj_grad(pts, c)[0]
        for it in range(20000):
            if time.time() > DEADLINE:
                break
            val, grad = _obj_grad(pts, c)
            gn = np.linalg.norm(grad)
            if gn < 1e-9:
                break
            direction = grad / gn
            # backtracking line search
            alpha = step
            ok = False
            for _ in range(18):
                cand = pts + alpha * direction
                v2, _ = _obj_grad(cand, c)
                if v2 > cur + 1e-12 * alpha * gn:
                    ok = True
                    break
                alpha *= 0.5
            if ok:
                pts = cand
                cur = v2
                step = min(step * 1.05, 0.5)
            else:
                step *= 0.5
                if step < 1e-5:
                    break
        # evaluate with exact score
        ev = _score(pts)
        if ev > best_val:
            best_val = ev
            best_pts = pts.copy()

    # final guard: ensure a valid, finite result
    if not np.all(np.isfinite(best_pts)):
        best_pts = _regular_octagon()
    # keep coordinates well within |coord| < 1e6 (they are ~O(1))
    return [[float(x), float(y)] for (x, y) in best_pts]
