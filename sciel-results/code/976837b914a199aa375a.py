# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

_DEADLINE = time.time() + 620.0  # safety margin before 630s kill


def _min_triangle_area(points):
    """Smallest triangle area over all triples. Vectorized for speed."""
    n = points.shape[0]
    # Use the shoelace cross-product form: area = 0.5*|cross(p_j-p_i, p_k-p_i)|
    # Build index arrays for all triples i<j<k
    i = np.repeat(np.arange(n), (n - 1) * (n - 2) // 2)
    # For each i, j runs over i+1..n-1 and k over j+1..n-1
    js = []
    ks = []
    for ii in range(n):
        jj = np.arange(ii + 1, n)
        for jjj in jj:
            kk = np.arange(jjj + 1, n)
            js.append(np.full(len(kk), jjj))
            ks.append(kk)
    j = np.concatenate(js)
    k = np.concatenate(ks)
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.min(np.abs(cross))


def _score(points):
    """A_min / A_hull, or 0.0 if invalid."""
    if points.shape != (13, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    A_min = _min_triangle_area(points)
    if A_min <= 0.0:
        return 0.0
    return A_min / A_hull


def _objective(x):
    """Negative score for minimization. Points are x[0:26] reshaped to (13,2)."""
    pts = x.reshape(13, 2)
    s = _score(pts)
    return -s


def _nelder_mead(pts, maxiter=3000, tol=1e-9):
    """Local Nelder-Mead refinement. Returns improved points."""
    from scipy.optimize import minimize
    x0 = pts.reshape(-1)
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': tol,
                                'fatol': tol, 'adaptive': True})
        new = res.x.reshape(13, 2)
        if np.all(np.isfinite(new)) and _score(new) >= _score(pts) - 1e-12:
            return new
    except Exception:
        pass
    return pts


def _powell(pts, maxiter=2000):
    """Powell refinement."""
    from scipy.optimize import minimize
    x0 = pts.reshape(-1)
    try:
        res = minimize(_objective, x0, method='Powell',
                       options={'maxiter': maxiter, 'ftol': 1e-10,
                                'xtol': 1e-9})
        new = res.x.reshape(13, 2)
        if np.all(np.isfinite(new)) and _score(new) >= _score(pts) - 1e-12:
            return new
    except Exception:
        pass
    return pts


def _sa_refine(pts, iters=4000, rng=None):
    """Simulated-annealing style perturbation search."""
    if rng is None:
        rng = np.random.default_rng(0)
    best = pts.copy()
    best_s = _score(best)
    cur = pts.copy()
    cur_s = _score(cur)
    T = 1e-3
    T_min = 1e-8
    for it in range(iters):
        if time.time() > _DEADLINE - 5:
            break
        step = T * (2.0 + 0.5 * np.sin(it * 0.01))
        cand = cur + rng.normal(0.0, step, size=(13, 2))
        cs = _score(cand)
        if cs >= cur_s or (cs > 0 and rng.random() < np.exp((cs - cur_s) / max(T, 1e-12))):
            cur = cand
            cur_s = cs
            if cur_s > best_s:
                best = cur.copy()
                best_s = cur_s
        T *= 0.9995
        if T < T_min:
            T = T_min
    return best


def _init_concentric(rng):
    """Concentric polygons: outer ring of 7, middle 4, inner 2."""
    pts = []
    # outer heptagon
    for k in range(7):
        ang = 2 * np.pi * k / 7 + rng.normal(0, 0.05)
        r = 1.0 + rng.normal(0, 0.02)
        pts.append([r * np.cos(ang), r * np.sin(ang)])
    # middle square
    for k in range(4):
        ang = 2 * np.pi * k / 4 + np.pi / 4 + rng.normal(0, 0.05)
        r = 0.55 + rng.normal(0, 0.03)
        pts.append([r * np.cos(ang), r * np.sin(ang)])
    # inner two
    for k in range(2):
        ang = 2 * np.pi * k / 2 + rng.normal(0, 0.1)
        r = 0.2 + rng.normal(0, 0.03)
        pts.append([r * np.cos(ang), r * np.sin(ang)])
    return np.array(pts)


def _init_fibonacci(rng):
    """Fibonacci (sunflower) lattice with jitter."""
    n = 13
    golden = (1 + np.sqrt(5)) / 2
    pts = []
    for k in range(n):
        r = np.sqrt((k + 0.5) / n)
        theta = 2 * np.pi * k / golden + rng.normal(0, 0.03)
        pts.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(pts)


def _init_grid(rng):
    """Jittered 4x4 grid with one corner removed, scaled."""
    pts = []
    coords = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            coords.append((i, j))
    for (i, j) in coords:
        x = i + rng.normal(0, 0.15)
        y = j + rng.normal(0, 0.15)
        pts.append([x, y])
    arr = np.array(pts)
    # center and scale
    arr -= arr.mean(axis=0)
    arr /= np.max(np.abs(arr))
    return arr


def _init_random(rng):
    return rng.uniform(-1, 1, size=(13, 2))


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region maximizing the
    smallest triangle area relative to the convex hull area.

    Returns:
        points: np.ndarray of shape (13,2).
    """
    rng = np.random.default_rng(2024)
    inits = [
        _init_concentric(rng),
        _init_fibonacci(rng),
        _init_grid(rng),
        _init_random(rng),
    ]
    # add a few more diverse structured starts
    for seed in (7, 13, 99, 123):
        r2 = np.random.default_rng(seed)
        inits.append(_init_concentric(r2))
        inits.append(_init_fibonacci(r2))
        inits.append(_init_grid(r2))

    best = None
    best_s = -1.0
    for init in inits:
        if time.time() > _DEADLINE - 30:
            break
        s = _score(init)
        if s > best_s:
            best = init.copy()
            best_s = s

    # Local refinement on the best, and on each promising init with time budget
    for init in inits:
        if time.time() > _DEADLINE - 20:
            break
        s = _score(init)
        if s <= best_s * 0.9:
            continue  # not promising enough
        # Nelder-Mead
        refined = _nelder_mead(init)
        rs = _score(refined)
        if rs > best_s:
            best = refined.copy()
            best_s = rs
        # Powell
        refined2 = _powell(refined)
        r2s = _score(refined2)
        if r2s > best_s:
            best = refined2.copy()
            best_s = r2s
        # SA polish
        refined3 = _sa_refine(refined2, iters=2000, rng=rng)
        r3s = _score(refined3)
        if r3s > best_s:
            best = refined3.copy()
            best_s = r3s

    # Final polish on the global best
    if time.time() < _DEADLINE - 15:
        final = _nelder_mead(best, maxiter=5000)
        fs = _score(final)
        if fs > best_s:
            best = final
            best_s = fs
    if time.time() < _DEADLINE - 10:
        final2 = _sa_refine(best, iters=3000, rng=rng)
        f2s = _score(final2)
        if f2s > best_s:
            best = final2
            best_s = f2s

    if best is None or best_s <= 0:
        # fallback: pure random
        best = rng.uniform(-1, 1, size=(13, 2))

    return best


# EVOLVE-BLOCK-END
