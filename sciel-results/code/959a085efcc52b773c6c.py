import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE_MARGIN = 12.0  # seconds to spare before the 150 s kill


def construct():
    start_time = time.time()
    time_limit = 150.0 - DEADLINE_MARGIN

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)  # 286

    def compute_score(points):
        """min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        return float(np.min(areas)) / area_hull

    def _normalize_points(points):
        """Center and scale so that the convex-hull area is 1."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _areas_only(points_flat):
        """Return the 286 triangle areas for a normalized point set."""
        pts = _normalize_points(points_flat.reshape(-1, 2))
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        return 0.5 * cross

    def _soft_objective(points_flat, beta=60.0):
        """Smooth approximation to -min area (for gradient methods)."""
        try:
            areas = _areas_only(points_flat)
        except Exception:
            return 1e10
        neg_b = -beta * areas
        m = np.max(neg_b)
        if not np.isfinite(m) or m < -1e6:
            return 1e10
        lse = m + np.log(np.sum(np.exp(neg_b - m)))
        return -lse / beta

    def _hard_objective(points_flat):
        """Negative min area (for derivative-free polish)."""
        try:
            areas = _areas_only(points_flat)
            m = float(np.min(areas))
        except Exception:
            return 1e10
        if not np.isfinite(m):
            return 1e10
        return -m

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n <= 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)])

    def _time_left():
        return start_time + time_limit - time.time()

    # ------------------------------------------------------------------
    # 1.  Build a broad, diverse set of seeds (symmetric families + noise)
    # ------------------------------------------------------------------
    seeds = []

    # Regular 13-gon (outer only)
    seeds.append(_regular_polygon(13, 1.0))

    # k-outer + (13-k)-inner regular polygons (k from 3..12)
    for k_outer in range(3, 13):
        k_inner = N - k_outer
        for r in [0.2, 0.35, 0.5, 0.65, 0.8]:
            outer = _regular_polygon(k_outer, 1.0)
            if k_inner == 1:
                inner = np.array([[0.0, 0.0]])
            else:
                inner = _regular_polygon(k_inner, r)
                # slight rotation offset for better coverage
                inner_angles = np.arctan2(inner[:, 1], inner[:, 0]) + np.pi / k_inner
                inner = np.column_stack([r * np.cos(inner_angles), r * np.sin(inner_angles)])
            seeds.append(np.vstack([outer, inner]))

    # Random initialisations
    rng = np.random.RandomState(7)
    for _ in range(40):
        r = rng.uniform(0.1, 1.0, size=(N,))
        th = rng.uniform(0, 2 * np.pi, size=N)
        r = np.sort(r)[::-1]  # ensure hull variety: some points well outside
        seeds.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    # ------------------------------------------------------------------
    # 2.  Coarse Nelder-Mead global pass over all seeds
    #     (fast: 500 iters × 13 dims; keeps best point per seed)
    # ------------------------------------------------------------------
    best_points = seeds[0].copy()
    best_score = -float('inf')

    for idx, seed in enumerate(seeds):
        if _time_left() < 40.0:   # need enough time for the gradient phase
            break
        try:
            res = opt.minimize(
                _soft_objective,
                seed.flatten(),
                args=(30.0,),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8,
                         'adaptive': True}
            )
            cand = res.x.reshape(-1, 2)
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3.  Gradient polishing: L-BFGS-B with annealed soft-min (beta)
    #     Multiple passes on the current best AND the top-k seeds.
    # ------------------------------------------------------------------
    top_seeds = [seed.copy() for seed in seeds[:8]]
    # Sort seeds by a quick Nelder-Mead score (reuse results)
    scored_seeds = []
    for idx, seed in enumerate(seeds):
        try:
            res = opt.minimize(_soft_objective, seed.flatten(), args=(30.0,),
                               method='Nelder-Mead', options={'maxiter': 80,
                                                               'adaptive': True})
            s = compute_score(res.x.reshape(-1, 2))
            scored_seeds.append((s, res.x.reshape(-1, 2)))
        except Exception:
            pass
    scored_seeds.sort(reverse=True, key=lambda t: t[0])

    for idx in range(min(6, len(scored_seeds))):
        if _time_left() < 20.0:
            break
        current = scored_seeds[idx][1].copy()
        for beta in [40.0, 100.0, 200.0, 400.0]:
            if _time_left() < 5.0:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12,
                             'maxfun': 2000}
                )
                current = res.x.reshape(-1, 2)
                s = compute_score(current)
                if s > best_score:
                    best_score = s
                    best_points = current.copy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 4.  Derivative-free micro-polish on the global best (Powell)
    # ------------------------------------------------------------------
    if _time_left() > 3.0:
        try:
            res = opt.minimize(
                _hard_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-14, 'ftol': 1e-14,
                         'maxfun': 20000}
            )
            cand = res.x.reshape(-1, 2)
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 5.  Final hard local search: alternate Powell restarts from
    #     slightly perturbed versions of the best point
    # ------------------------------------------------------------------
    if _time_left() > 3.0:
        rng2 = np.random.RandomState(123)
        for perturb_scale in [0.05, 0.02, 0.008]:
            if _time_left() < 2.0:
                break
            perturbed = best_points + rng2.randn(N, 2) * perturb_scale
            try:
                res = opt.minimize(
                    _hard_objective,
                    perturbed.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-14, 'ftol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    return best_points
