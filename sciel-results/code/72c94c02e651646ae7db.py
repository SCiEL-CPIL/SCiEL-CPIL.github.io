"""Packing 50 equal spheres in a unit cube: structured init + smooth-min gradient ascent."""
import numpy as np
import time

N = 50
DEADLINE = 140.0   # seconds wall-clock safety margin under the 150s kill

def _radius(c):
    """Hard packing radius: min over pairs of half-distance and over walls of clearance."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = 0.5 * np.min(dist)
    r_wall = np.min(0.5 - np.abs(c))
    return float(min(r_pair, r_wall))


def _softmin(x, beta):
    """Smooth approximation of min(x); returns (value, dvalue/dx)."""
    m = np.min(x)
    s = np.sum(np.exp(-beta * (x - m)))
    v = m - np.log(s) / beta
    w = np.exp(-beta * (x - m)) / s
    return v, w


def construct():
    t0 = time.time()

    # ---- Initialization: 5x5x2 grid (symmetric, non-overlapping) ----
    x = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    z = np.array([-0.25, 0.25])
    pts = np.array([(a, b, c) for c in z for b in x for a in x])
    c = pts[:N].astype(float).copy()
    c0 = c.copy()

    r_best = _radius(c)
    c_best = c.copy()
    beta = 30.0

    # ---- Gradient ascent on the smooth-min objective ----
    step = 0.0005
    for it in range(200000):
        if time.time() - t0 > DEADLINE:
            break

        d = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair = 0.5 * dist                       # pairwise half-distances
        wall = 0.5 - np.abs(c)                  # (N,3) wall clearances

        v_pair, w_pair = _softmin(pair, beta)
        v_wall, w_wall = _softmin(wall, beta)

        # objective = min(v_pair, v_wall) - eps*||c-c0||^2
        if v_pair <= v_wall:
            a, b = 1.0, 0.0
        else:
            a, b = 0.0, 1.0

        grad = np.zeros_like(c)
        # pair contribution: d(v_pair)/d c_i = a * sum_j w_ij * 0.5 * (c_i - c_j)/|c_i-c_j|
        wsum = np.sum(w_pair, axis=1)
        for i in range(N):
            diff = c[i] - c
            unit = diff / dist[i]
            contrib = w_pair[i] * unit
            contrib[i] = 0.0
            grad[i] = a * 0.5 * np.sum(contrib, axis=0)
        # wall contribution: d(v_wall)/d c_ik = b * w_wall[i,k] * sign(c_ik)
        sgn = np.sign(c)
        grad += b * w_wall * sgn
        # L2 regularizer
        grad -= 0.001 * (c - c0)

        c = c + step * grad
        c = np.clip(c, -0.5, 0.5)

        # anneal: lower beta over time, shrink step slowly
        beta = max(8.0, beta * 0.9995)
        step = max(1e-5, step * 0.9999)

        r = _radius(c)
        if r > r_best:
            r_best = r
            c_best = c.copy()

    return c_best
