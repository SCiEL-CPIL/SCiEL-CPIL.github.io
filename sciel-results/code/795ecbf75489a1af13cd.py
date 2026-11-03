"""Rewritten: hex-lattice init + vectorized soft-force relaxation + local polish."""
import time
import numpy as np

N = 200
HALF = 0.5


def _score(pts):
    """Largest common radius (higher better)."""
    if pts is None or pts.shape != (N, 2):
        return 0.0
    p = np.asarray(pts, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        return 0.0
    # wall distance
    wall = np.min(np.minimum(np.minimum(HALF - p[:, 0], p[:, 0] + HALF),
                             np.minimum(HALF - p[:, 1], p[:, 1] + HALF)))
    # pairwise min distance
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    r = min(wall, np.min(d) * 0.5)
    return r if r > 0 else 0.0


def _hex_init(n):
    """Approximately n points on a hexagonal lattice, scaled to the unit square."""
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    pts = []
    for r in range(rows):
        y = r * 1.5
        off = 0.5 if (r % 2) else 0.0
        for c in range(rows + 1):
            pts.append((c + off, y))
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n], dtype=np.float64)
    # scale so the point cloud fits inside [-0.5,0.5]^2 with a small margin
    span = np.max(pts, axis=0) - np.min(pts, axis=0)
    scale = 0.9 / np.max(span)
    pts = (pts - np.mean(pts, axis=0)) * scale
    return pts


def _soft_step(pts, r, w, niter, k):
    """Vectorized force-directed relaxation: repulsion + containment."""
    p = pts.copy()
    for it in range(niter):
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        # repulsion force for pairs closer than 2r
        f = np.zeros_like(p)
        mask = d < 2.0 * r
        for i in range(len(p)):
            j = np.where(mask[i] & (d[i] > 0))[0]
            if len(j) == 0:
                continue
            diff = p[i] - p[j]
            f[i] += np.sum(diff / d[i][j][:, None], axis=0)
        # containment force (push back toward center when outside margin)
        for i in range(len(p)):
            for axis in range(2):
                if p[i, axis] > HALF - r:
                    f[i, axis] -= (p[i, axis] - (HALF - r))
                if p[i, axis] < -HALF + r:
                    f[i, axis] += ((-HALF + r) - p[i, axis])
        p = p + w * f
        # clamp to square
        p = np.clip(p, -HALF, HALF)
    return p


def _polish(pts, r, niter, step):
    """Local greedy: move the most-constrained point toward a better position."""
    p = pts.copy()
    for it in range(niter):
        # find the point with the smallest clearance
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        wall = np.min(np.minimum(np.minimum(HALF - p[:, 0], p[:, 0] + HALF),
                                 np.minimum(HALF - p[:, 1], p[:, 1] + HALF)))
        clearance = np.minimum(d.min(axis=1) * 0.5, wall)
        i = int(np.argmin(clearance))
        # try small moves in 8 directions
        best = p[i].copy()
        bestc = clearance[i]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in dirs:
            cand = p[i] + step * np.array([dx, dy])
            cand = np.clip(cand, -HALF, HALF)
            # check against all others and wall
            dd = np.linalg.norm(p - cand)
            dd[i] = np.inf
            dw = np.min(np.minimum(np.minimum(HALF - cand[0], cand[0] + HALF),
                                   np.minimum(HALF - cand[1], cand[1] + HALF)))
            c = min(dd.min() * 0.5, dw)
            if c > bestc + 1e-12:
                bestc = c
                best = cand
        p[i] = best
    return p


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    best = _hex_init(N)
    best_r = _score(best)

    # Phase 1: soft relaxation to find a good radius estimate
    r0 = best_r
    for r in [r0 * 1.02, r0 * 1.05, r0 * 1.08, r0 * 1.12, r0 * 1.18]:
        if time.time() > deadline - 40:
            break
        cand = _soft_step(best, r, 0.004, 400, 0)
        rc = _score(cand)
        if rc > best_r:
            best_r = rc
            best = cand

    # Phase 2: local polish
    if time.time() < deadline - 10:
        cand = _polish(best, best_r, 3000, 0.002)
        rc = _score(cand)
        if rc > best_r:
            best_r = rc
            best = cand

    # Phase 3: one more relaxation pass at the discovered radius
    if time.time() < deadline - 5:
        cand = _soft_step(best, best_r, 0.002, 300, 0)
        rc = _score(cand)
        if rc > best_r:
            best_r = rc
            best = cand

    # Final validity guard
    best = np.clip(best, -HALF, HALF)
    return best
