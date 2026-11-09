# EVOLVE-BLOCK-START
import time
import numpy as np

_TIME_LIMIT = 630.0
_DEADLINE = time.time() + _TIME_LIMIT - 30.0  # 30 s safety margin before the 630 s kill


def _normalize(poly):
    """Subtract centroid and divide by the mean distance to it."""
    poly = np.asarray(poly, dtype=float)
    c = poly.mean(axis=0)
    p = poly - c
    d = np.linalg.norm(p, axis=1)
    m = float(d.mean())
    if m <= 0.01:
        return None
    return p / m


def _is_strictly_convex(poly):
    """Return True if the polygon is strictly convex (all turns same nonzero sign)."""
    n = poly.shape[0]
    p = poly
    signs = []
    for i in range(n):
        a = p[i]
        b = p[(i + 1) % n]
        c = p[(i + 2) % n]
        e1 = b - a
        e2 = c - b
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(cross) < 1e-12:
            return False
        signs.append(1 if cross > 0 else -1)
    if len(set(signs)) != 1:
        return False
    # total turning must be +-2*pi (for a simple strictly convex polygon this holds)
    return True


def _vertex_scores(poly):
    """Return per-vertex scores for the given (n,2) polygon."""
    n = poly.shape[0]
    # pairwise squared distances
    diff = poly[:, None, :] - poly[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    scores = np.zeros(n)
    for i in range(n):
        dists = np.sort(d[i])  # d_0 .. d_18, d_0 = 0
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = dists[1:] - dists[:-1]  # d_{k+1} - d_k for k=0..17
        k = int(np.argmin(gaps))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            sel = dists[0:4]
        elif k == 17:
            sel = dists[15:19]
        else:
            sel = dists[k - 1:k + 3]
        ratios = np.maximum(sel / D, D / sel)
        raw = float(ratios.mean())
        scores[i] = min(1.0, 1.0 / raw)
    return scores


def _min_score(poly):
    s = _vertex_scores(poly)
    return float(s.min())


def _refine(poly, max_iter=60000):
    """Coordinate ascent on the minimum vertex score."""
    poly = np.array(poly, dtype=float)
    n = poly.shape[0]
    cur = _min_score(poly)
    step = 0.05
    best = cur
    best_poly = poly.copy()
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        # pick the worst vertex (ties broken by index)
        scores = _vertex_scores(poly)
        i = int(np.argmin(scores))
        improved = False
        for dx in (-step, step):
            for dy in (-step, step):
                q = poly.copy()
                q[i] += np.array([dx, dy])
                if not _is_strictly_convex(q):
                    continue
                val = _min_score(q)
                if val > cur + 1e-9:
                    poly = q
                    cur = val
                    improved = True
                    break
            if improved:
                break
        if improved:
            if cur > best:
                best = cur
                best_poly = poly.copy()
            step = min(step * 1.15, 0.3)
        else:
            step = max(step * 0.7, 0.005)
            if step <= 0.0051:
                break
    return best_poly, best


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a strictly-convex 20-gon maximizing the minimum per-vertex equidistance score.

    Strategy: seed from the regular polygon, then run coordinate ascent that directly
    optimizes the task's minimum-vertex-score metric. A hard deadline guard returns the
    best polygon found before the wall-clock limit.
    """
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    # normalize the seed so it is well-scaled for the metric
    seed = _normalize(base)
    if seed is None:
        return base

    # First pass: refine the seed
    poly, best = _refine(seed)

    # Second pass: try a few perturbed seeds to escape shallow local minima
    rng = np.random.default_rng(12345)
    for trial in range(6):
        if time.time() > _DEADLINE - 10.0:
            break
        ang = angles + rng.uniform(-0.06, 0.06, size=n)
        rad = radius * (1.0 + rng.uniform(-0.08, 0.08, size=n))
        cand = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        cand = _normalize(cand)
        if cand is None or not _is_strictly_convex(cand):
            continue
        p2, b2 = _refine(cand)
        if b2 > best:
            best = b2
            poly = p2

    # Final normalization for output (does not change the metric, keeps it finite)
    out = _normalize(poly)
    if out is None:
        out = poly
    return out


# EVOLVE-BLOCK-END
