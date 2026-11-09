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


def _polish(poly, max_iter=30000):
    """Fine-grained coordinate ascent with small steps for final refinement."""
    poly = np.array(poly, dtype=float)
    cur = _min_score(poly)
    step = 0.01
    best = cur
    best_poly = poly.copy()
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        scores = _vertex_scores(poly)
        i = int(np.argmin(scores))
        improved = False
        # Try 8 directions (N, S, E, W, and 4 diagonals)
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                if dx == 0.0 and dy == 0.0:
                    continue
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
            step = min(step * 1.05, 0.05)
        else:
            step = max(step * 0.8, 0.001)
            if step <= 0.0011:
                break
    return best_poly, best


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a strictly-convex 20-gon maximizing the minimum per-vertex equidistance score.

    Strategy: 
    1. Generate multiple diverse seeds (regular polygon + perturbations).
    2. For each seed, run a two-phase optimization:
       - Phase 1: Coarse coordinate ascent to escape local minima.
       - Phase 2: Fine-grained polish pass for precision.
    3. Return the best polygon found within the time budget.
    """
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # Generate candidate seeds
    seeds = []
    
    # Seed 0: Regular polygon
    base = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    seed = _normalize(base)
    if seed is not None and _is_strictly_convex(seed):
        seeds.append(seed)

    # Seeds 1-7: Perturbed regular polygons
    rng = np.random.default_rng(42)
    for trial in range(7):
        ang = angles + rng.uniform(-0.1, 0.1, size=n)
        rad = radius * (1.0 + rng.uniform(-0.15, 0.15, size=n))
        cand = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        cand = _normalize(cand)
        if cand is not None and _is_strictly_convex(cand):
            seeds.append(cand)

    # Evaluate each seed with two-phase optimization
    best_poly = None
    best_score = -1.0

    for idx, seed in enumerate(seeds):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Phase 1: Coarse refinement
        poly1, score1 = _refine(seed, max_iter=20000)
        
        # Phase 2: Fine polish
        poly2, score2 = _polish(poly1, max_iter=15000)
        
        if score2 > best_score:
            best_score = score2
            best_poly = poly2

    # Final normalization
    if best_poly is not None:
        out = _normalize(best_poly)
        if out is not None:
            return out
        return best_poly
    
    # Fallback: return regular polygon
    base = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return base


# EVOLVE-BLOCK-END
