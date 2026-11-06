"""Packing 50 equal circles in the unit disk: repulsion simulation + polish."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
DEADLINE = 150.0          # hard wall-clock limit
MARGIN = 8.0              # return this many seconds before the limit
RNG_SEED = 12345


def _score(pts):
    """Score = min(min pairwise half-distance, min room to boundary)."""
    pts = np.asarray(pts)
    d = cKDTree(pts).query(pts, k=2)[0][:, 1]
    return float(min(d.min() * 0.5, 1.0 - np.linalg.norm(pts, axis=1).max()))


def _project(pts):
    """Push every centre back inside the unit disk."""
    r = np.linalg.norm(pts, axis=1)
    over = r > 1.0
    if over.any():
        pts[over] *= (1.0 / r[over])[:, None]
    return pts


def _repulsion(pts, it):
    """One repulsion step: push apart too-close pairs + boundary pressure."""
    r = np.linalg.norm(pts, axis=1)
    force = np.zeros_like(pts)
    d = cKDTree(pts).query(pts, k=2)[0][:, 1]
    too = d < 0.02
    if too.any():
        idx = np.where(too)[0]
        for i in idx:
            j = int(cKDTree(pts).query(pts[i], k=2)[1][i])
            vec = pts[i] - pts[j]
            n = np.linalg.norm(vec)
            if n > 1e-12:
                force[i] += vec / n
    # boundary: pull inward proportionally to how close to the wall
    close = r > 0.985
    if close.any():
        force[close] -= pts[close] / np.maximum(r[close], 1e-9)[:, None] * 0.5
    step = 0.004 * (1.0 / (1.0 + it * 0.002))
    pts = _project(pts + step * force)
    return pts


def _repel_sim(pts, iters):
    pts = _project(pts.copy())
    for it in range(iters):
        pts = _repulsion(pts, it)
    return pts


def _init_rings():
    """Concentric rings with alternating angular offsets (good diversity)."""
    rings = [(1, 0.0), (6, 0.32), (12, 0.55), (18, 0.76), (13, 0.92)]
    pts = []
    for count, rad in rings:
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            th = np.linspace(0, 2 * np.pi, count, endpoint=False)
            pts.append(np.column_stack([rad * np.cos(th), rad * np.sin(th)]))
    pts = np.vstack(pts)[:N]
    return _project(pts)


def _init_hex():
    """Hexagonal lattice clipped to the disk."""
    pts = []
    s = 0.2
    for j in range(-12, 13):
        for i in range(-12, 13):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 0.95 * 0.95:
                pts.append([x, y])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    return _project(pts[order][:N])


def _init_random():
    pts = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x * x + y * y <= 0.95 * 0.95:
                pts.append([x, y])
                break
    return _project(np.array(pts))


def _polish(pts, iters=3000):
    """Local optimum: move the most-constrained circle to its best local spot."""
    pts = pts.copy()
    base = _score(pts)
    for _ in range(iters):
        d = cKDTree(pts).query(pts, k=2)[0][:, 1]
        room = 1.0 - np.linalg.norm(pts, axis=1)
        minr = np.minimum(d * 0.5, room)
        i = int(np.argmin(minr))
        # try small moves around i
        best = pts[i].copy()
        best_s = minr[i]
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for mag in (0.002, 0.005, 0.01, 0.02):
                cand = pts[i].copy()
                cand += mag * np.array([np.cos(ang), np.sin(ang)])
                if np.linalg.norm(cand) > 1.0:
                    continue
                pts[i] = cand
                dd = cKDTree(pts).query(pts[i], k=2)[0][1]
                s = min(dd, 1.0 - np.linalg.norm(cand))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
        pts[i] = best
    return pts


def construct():
    t0 = time.time()
    np.random.seed(RNG_SEED)

    inits = [_init_rings(), _init_hex()]
    for _ in range(3):
        inits.append(_init_random())

    best = None
    best_s = -1.0
    for init in inits:
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        for iters in (400, 2000):
            if time.time() - t0 > DEADLINE - MARGIN:
                break
            cand = _repel_sim(init, iters)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand

    # final polish on the best so far
    if best is not None and time.time() - t0 < DEADLINE - MARGIN:
        polished = _polish(best, iters=2000)
        if _score(polished) >= best_s:
            best = polished

    if best is None:
        best = _init_rings()
    return best
