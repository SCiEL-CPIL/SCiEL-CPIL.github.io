"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 145  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to soft_min_area(pts) <= t.
    x = [t, pts_flat...] where t is at index 0.
    """
    t = x[0]
    pts = x[1:].reshape(-1, 2)
    pts = _project_to_triangle(pts)
    sm = _soft_min_area(pts, beta)
    # Penalty for violating t >= soft_min
    violation = max(0.0, sm - t)
    return t + 1000.0 * violation ** 2


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Initial configuration: corners, edge midpoints, centroid, and interior points ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [1/3, 0.0],
        [2/3, 0.0],
        [0.5, S3H/3],
        [0.5, 2*S3H/3],
    ], dtype=float)
    init = _project_to_triangle(init)
    _update_best(init)

    # ---- Multi-start bases with varying perturbation ----
    bases = [init.copy()]
    for _ in range(20):
        pert = init + rng.normal(0, 0.2, size=(N, 2))
        pert = _project_to_triangle(pert)
        bases.append(pert)

    def _optimize_soft(pts0, beta, maxiter=300):
        """Optimize using L-BFGS-B on the soft-min objective."""
        def objective(x):
            pts = x.reshape(-1, 2)
            pts = _project_to_triangle(pts)
            return -_soft_min_area(pts, beta)

        try:
            res = minimize(objective, pts0.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            pts = res.x.reshape(-1, 2)
            pts = _project_to_triangle(pts)
            return pts
        except Exception:
            return pts0

    def _optimize_nm(pts0, maxiter=5000):
        """Nelder-Mead polish on the hard min objective."""
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)

        try:
            res = minimize(nm_obj, pts0.flatten(), method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return p
        except Exception:
            return pts0

    def _optimize_epigraph(pts0, beta, maxiter=200):
        """Epigraph optimization: optimize t and pts jointly."""
        x0 = np.concatenate([[best_area if best_pts is not None else 0.01], pts0.flatten()])

        def obj(x):
            t = x[0]
            pts = x[1:].reshape(-1, 2)
            pts = _project_to_triangle(pts)
            sm = _soft_min_area(pts, beta)
            violation = max(0.0, sm - t)
            return t + 500.0 * violation ** 2

        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            pts = res.x[1:].reshape(-1, 2)
            pts = _project_to_triangle(pts)
            return pts
        except Exception:
            return pts0

    # ---- Phase A: Multi-start soft-min gradient optimization (annealed beta) ----
    n_starts = 28
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases[s % len(bases)]
        pts = base + rng.normal(0, 0.05, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)

        for beta in [15.0, 40.0, 100.0, 250.0, 600.0]:
            if not _elapsed_ok(35):
                break
            pts = _optimize_soft(pts, beta, maxiter=200)
            _update_best(pts)

    # ---- Phase B: Epigraph refinement on top candidates ----
    if best_pts is not None and _elapsed_ok(25):
        # Refine the best with epigraph
        for beta in [50.0, 150.0, 400.0]:
            if not _elapsed_ok(20):
                break
            cand = best_pts + rng.normal(0, 0.02, size=(N, 2))
            cand = _project_to_triangle(cand)
            cand = _optimize_epigraph(cand, beta, maxiter=150)
            _update_best(cand)

    # ---- Phase C: Nelder-Mead local polish on the best configuration ----
    if best_pts is not None and _elapsed_ok(15):
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(10):
                break
            for _ in range(8):
                if not _elapsed_ok(8):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)
                cand = _optimize_nm(cand, maxiter=4000)
                _update_best(cand)

    # ---- Phase D: High-precision final polish ----
    if best_pts is not None and _elapsed_ok(5):
        # Very tight Nelder-Mead
        cand = _optimize_nm(best_pts, maxiter=10000)
        _update_best(cand)

        # One more round of tiny perturbations + NM
        if _elapsed_ok(3):
            for _ in range(5):
                if not _elapsed_ok(2):
                    break
                cand = best_pts + rng.normal(0, 0.003, size=(N, 2))
                cand = _project_to_triangle(cand)
                cand = _optimize_nm(cand, maxiter=6000)
                _update_best(cand)

    if best_pts is None:
        # fallback random feasible configuration
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            a = 1 - u - v
            x = a * 0 + u * 1 + v * 0.5
            y = a * 0 + u * 0 + v * S3H
            pts[i] = [x, y]
        return pts

    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
