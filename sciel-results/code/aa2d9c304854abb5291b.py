"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_DEADLINE = time.time() + 143.0  # explicit wall-clock guard (150s kill, 7s safety margin)

def _time_ok(margin=0.0):
    return time.time() < _DEADLINE - margin

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        if m < 2:
            continue
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            local_min = 0.5 * np.min(np.abs(crosses))
            if local_min < min_area:
                min_area = local_min
    return min_area

def _soft_min_triangle_area(pts, alpha=60.0):
    """
    Smooth approximation of the minimum triangle area via a log-sum-exp of
    the smallest areas (annealable through alpha).
    """
    n = len(pts)
    if n < 3:
        return 0.0
    # gather all triangle areas into one array
    areas = []
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            areas.append(0.5 * np.abs(crosses))
    if not areas:
        return 0.0
    a = np.concatenate(areas)
    a = a[a > 1e-12]
    if len(a) == 0:
        return 0.0
    amin = np.min(a)
    shifted = -alpha * (a - amin)
    # numerical stability: shift by max
    shifted = shifted - np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted))) + np.max(shifted) - alpha * amin
    return -log_sum / alpha

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _neg_smooth_score(params, alpha=60.0):
    pts = params.reshape(-1, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    return -(_soft_min_triangle_area(pts, alpha) / hull_a)

def _build_from_rings(ring_params):
    """
    Build N points from a list of (count, radius, phase) tuples.
    Each ring places `count` evenly spaced points at `radius` with angular offset `phase`.
    """
    pts = []
    for count, radius, phase in ring_params:
        if count == 0:
            continue
        if count == 1:
            pts.append(np.array([[radius * np.cos(phase), radius * np.sin(phase)]]))
        else:
            angles = phase + 2 * np.pi * np.arange(count) / count
            pts.append(radius * np.column_stack([np.cos(angles), np.sin(angles)]))
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def _ring_params_to_points(params, ring_counts):
    """params: interleaved [r0, p0, r1, p1, ...] for each ring."""
    ring_params = []
    for i, count in enumerate(ring_counts):
        ring_params.append((count, abs(params[2 * i]), params[2 * i + 1]))
    return _build_from_rings(ring_params)

def _neg_ring_score(params, ring_counts):
    pts = _ring_params_to_points(params, ring_counts)
    return -_score(pts)

def _neg_ring_smooth_score(params, ring_counts, alpha=60.0):
    pts = _ring_params_to_points(params, ring_counts)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    return -(_soft_min_triangle_area(pts, alpha) / hull_a)

def _anneal_ring(ring_counts, x0, max_iter=120):
    """Progressive-annealing L-BFGS-B on the smooth ring objective, then exact NM polish."""
    pts = _ring_params_to_points(x0, ring_counts)
    cur_score = _score(pts)
    cur_params = x0.copy()
    # anneal alpha from low (broad) to high (sharp)
    for alpha in [15.0, 40.0, 90.0, 200.0]:
        if not _time_ok(8.0):
            break
        try:
            res = minimize(
                lambda p, a=alpha: _neg_ring_smooth_score(p, ring_counts, a),
                cur_params,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            cand_pts = _ring_params_to_points(res.x, ring_counts)
            cand_score = _score(cand_pts)
            if cand_score >= cur_score:
                cur_score = cand_score
                cur_params = res.x.copy()
        except Exception:
            pass
    # exact Nelder-Mead polish on the true objective
    if _time_ok(4.0):
        try:
            res = minimize(
                lambda p: _neg_ring_score(p, ring_counts),
                cur_params,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            cand_pts = _ring_params_to_points(res.x, ring_counts)
            cand_score = _score(cand_pts)
            if cand_score > cur_score:
                cur_score = cand_score
                cur_params = res.x.copy()
        except Exception:
            pass
    return _ring_params_to_points(cur_params, ring_counts), cur_score

def construct():
    rng = np.random.RandomState(42)

    # ---- 1. Diverse initializations ----
    inits = []
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles13), np.sin(angles13)]))
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    inits.append(np.vstack([np.column_stack([np.cos(angles12), np.sin(angles12)]), [[0.0, 0.0]]]))

    # ring-structure initializations (radius, phase) per ring
    ring_structures = [
        [13], [1, 12], [1, 6, 6], [7, 6], [2, 11], [3, 10],
        [4, 9], [5, 8], [6, 7], [1, 3, 9], [1, 4, 8], [1, 5, 7], [1, 2, 10],
    ]
    for rc in ring_structures:
        n_rings = len(rc)
        # base radii: spread from outer to inner
        base_r = [1.0 / (i + 1) for i in range(n_rings)]
        # a few phase patterns
        for ph in [0.0, np.pi / 6.0]:
            x0 = np.zeros(2 * n_rings)
            for i in range(n_rings):
                x0[2 * i] = base_r[i]
                x0[2 * i + 1] = ph if i > 0 else 0.0
            inits.append(_ring_params_to_points(x0, rc))

    # random points in unit disk
    for _ in range(4):
        r = np.sqrt(rng.rand(N))
        th = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    # jittered regular 13-gon
    for jitter in [0.02, 0.06]:
        rr = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([rr * np.cos(angles13), rr * np.sin(angles13)]))

    # ---- 2. Score initializations, keep best ----
    best_score = 0.0
    best_pts = inits[0].copy()
    scored = []
    for init_pts in inits:
        if not _time_ok(45.0):
            break
        s = _score(init_pts)
        scored.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    scored.sort(key=lambda x: -x[0])

    # ---- 3. Refine top free-form candidates (smooth -> exact) ----
    top = []
    for s, p in scored[:4]:
        if not any(np.allclose(p, q, atol=1e-5) for q in top):
            top.append(p)
    top.append(best_pts)

    for init_pts in top:
        if not _time_ok(12.0):
            break
        cur_pts = init_pts.copy()
        cur_score = _score(cur_pts)
        # smooth Powell
        if _time_ok(6.0):
            try:
                res = minimize(
                    lambda p: _neg_smooth_score(p, alpha=60.0),
                    cur_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                cs = _score(cand)
                if cs > cur_score:
                    cur_pts, cur_score = cand, cs
            except Exception:
                pass
        # exact NM
        if _time_ok(3.0):
            try:
                res = minimize(_neg_score, cur_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13})
                cand = res.x.reshape(-1, 2)
                cs = _score(cand)
                if cs > cur_score:
                    cur_pts, cur_score = cand, cs
            except Exception:
                pass
        if cur_score > best_score:
            best_score, best_pts = cur_score, cur_pts.copy()

    # ---- 4. Symmetric ring optimization with annealing ----
    for rc in ring_structures:
        if not _time_ok(10.0):
            break
        n_rings = len(rc)
        base_r = [1.0 / (i + 1) for i in range(n_rings)]
        x0_list = []
        x0_list.append(np.array([base_r[i] if i % 2 == 0 else base_r[i]
                                 for i in range(n_rings)] + [0.0] * n_rings))
        # random inits
        for _ in range(3):
            x0 = np.zeros(2 * n_rings)
            x0[0::2] = np.abs(rng.uniform(0.2, 1.2, size=n_rings))
            x0[1::2] = rng.uniform(-np.pi / 6, np.pi / 6, size=n_rings)
            x0_list.append(x0)
        for x0 in x0_list:
            if not _time_ok(6.0):
                break
            cand_pts, cand_score = _anneal_ring(rc, x0)
            if cand_score > best_score:
                best_score, best_pts = cand_score, cand_pts.copy()

    # ---- 5. Final high-precision polish on the best ----
    if _time_ok(5.0):
        try:
            res = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15})
            cand = res.x.reshape(-1, 2)
            cs = _score(cand)
            if cs > best_score:
                best_pts = cand
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
