"""Physics-based packing optimizer for 50 equal circles in a unit circle."""
import time
import numpy as np

N = 50
DEADLINE = 140.0  # seconds, safety margin below the 150 s wall-clock limit


def _min_pair(pts):
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min()


def _feasible_radius(p):
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return min(d.min() / 2.0, (1.0 - np.hypot(p[:, 0], p[:, 1])).min())


def _hex_init():
    best = None
    for base in np.linspace(0.32, 0.52, 41):
        s = 2.0 * base
        k = int(1.0 / s) + 2
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                pts.append((x, y))
        pts = np.array(pts)
        r = np.hypot(pts[:, 0], pts[:, 1])
        keep = r <= 1.0 - base
        pts = pts[keep]
        if len(pts) >= N:
            pts = pts[np.argsort(r, kind="stable")[:N]]
            r0 = _feasible_radius(pts)
            if best is None or r0 > best[0]:
                best = (r0, pts.copy())
    return best[1]


def construct():
    t0 = time.time()
    pts = _hex_init()
    best_r = _feasible_radius(pts)
    best_pts = pts.copy()

    K = 300.0
    WALL = 20000.0
    dt = 0.0025
    cur_floor = best_r * 0.98

    def step(p, dt):
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e-9)
        w = K / (d ** 3)
        np.fill_diagonal(w, 0.0)
        diff = p[:, None, :] - p[None, :, :]
        force = (w[..., None] * diff).sum(1)
        rad = np.hypot(p[:, 0], p[:, 1])
        near = np.clip(rad - (1.0 - 0.02), 0.0, None)
        wall_force = -WALL * near[..., None] * (p / np.maximum(rad, 1e-9)[..., None])
        return p + dt * (force + wall_force)

    def enforce_floor(p, fl):
        dmin = 2.0 * fl
        for _ in range(80):
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            bad = d < dmin
            if not bad.any():
                break
            diff = p[:, None, :] - p[None, :, :]
            for i in range(N):
                for j in range(i + 1, N):
                    if bad[i, j]:
                        vec = diff[i, j]
                        nrm = np.linalg.norm(vec)
                        if nrm < 1e-9:
                            vec = np.array([1.0, 0.0]); nrm = 1.0
                        push = (dmin - d[i, j]) * 0.5 * vec / nrm
                        p[i] = p[i] + push
                        p[j] = p[j] - push
        rad = np.hypot(p[:, 0], p[:, 1])
        over = rad > 1.0 - fl
        if over.any():
            p[over] = p[over] * ((1.0 - fl) / rad[over])[..., None]
        return p

    it = 0
    while time.time() - t0 < DEADLINE - 2:
        for _ in range(200):
            if time.time() - t0 > DEADLINE - 2:
                break
            pts = step(pts, dt)
            pts = enforce_floor(pts, cur_floor)
        r = _feasible_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            cur_floor = min(best_r * 0.995, best_r + 1e-4)
        else:
            if it % 10 == 0:
                cur_floor = min(best_r * 0.999, best_r + 1e-5)
                pts = best_pts.copy()
            dt *= 0.985
        it += 1

    # final polish
    for _ in range(300):
        if time.time() - t0 > DEADLINE - 1:
            break
        pts = best_pts.copy()
        r = _feasible_radius(pts)
        for _ in range(20):
            p = pts.copy()
            p[np.random.randint(N)] += np.random.randn(2) * 0.002
            p = enforce_floor(p, r * 0.999)
            rp = _feasible_radius(p)
            if rp > r:
                pts, r = p, rp
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # safety: ensure valid output
    if not (np.all(np.isfinite(best_pts)) and best_pts.shape == (N, 2)):
        best_pts = _hex_init()
    return best_pts
