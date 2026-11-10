"""Fresh design: hex seed + vectorized repulsion optimizer with deadline guard."""
import numpy as np
import time

N = 300


def _hex_seed(n):
    # Hexagonal lattice points inside the unit disk, sorted by distance from origin.
    best = None
    for s in np.linspace(0.05, 0.16, 130):
        k = int(1.0 / s) + 2
        xs = (np.arange(-k, k + 1) + 0.5 * (np.arange(-k, k + 1) % 2)) * s
        ys = np.arange(-k, k + 1) * s * np.sqrt(3.0) / 2.0
        X, Y = np.meshgrid(xs, ys)
        P = np.column_stack([X.ravel(), Y.ravel()])
        r = np.hypot(P[:, 0], P[:, 1])
        P = P[r <= 1.0 - s / 2.0]
        if len(P) >= n:
            order = np.argsort(r, kind="stable")
            P = P[order[:n]]
            dmin = 0.0
            for a in range(n):
                dd = np.hypot(*(P[a] - P[a + 1:]).T)
                if dd.size:
                    m = dd.min()
                    if m < dmin:
                        dmin = m
            score = min(dmin / 2.0, (1.0 - r[order[:n]].max()))
            if best is None or score > best[0]:
                best = (score, P.copy())
    return best[1]


def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    pts = _hex_seed(N)

    def score(P):
        d = np.hypot(*(P[:, None, :] - P[None, :, :]).T)
        np.fill_diagonal(d, np.inf)
        return min(d.min() / 2.0, 1.0 - np.hypot(P[:, 0], P[:, 1]).max())

    def step(P, lr, h):
        P = P.copy()
        d = np.hypot(*(P[:, None, :] - P[None, :, :]).T)
        np.fill_diagonal(d, np.inf)
        close = d < h
        d[~close] = np.inf
        d2 = np.minimum(d, h) ** 2
        d2[np.isinf(d2)] = np.inf
        diff = P[:, None, :] - P[None, :, :]
        force = np.einsum('ijk,ik->jk', diff / d2[:, :, None], close)
        P += lr * force
        r = np.hypot(P[:, 0], P[:, 1])
        m = r > 1.0
        if m.any():
            P[m] -= lr * 0.5 * (r[m] - 1.0)[:, None] * P[m] / np.maximum(r[m], 1e-9)[:, None]
        return P

    best = pts.copy()
    best_sc = score(pts)

    h = 2.5 * max(best_sc, 1e-4)
    lr = 0.05 * h
    it = 0
    while time.time() < deadline:
        P = step(best, lr, h)
        s = score(P)
        if s > best_sc:
            best_sc = s
            best = P.copy()
            h = max(2.0 * best_sc, 1e-4)
            lr = 0.05 * h
            it = 0
        else:
            it += 1
            if it >= 60:
                lr *= 0.85
                h *= 0.9
                it = 0
                if h < 1e-4:
                    break

    return best
