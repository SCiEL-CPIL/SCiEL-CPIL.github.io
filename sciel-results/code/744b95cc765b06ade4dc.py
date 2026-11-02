"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

New structural approach:
  1. Fast vectorized scoring over all 286 triples.
  2. A "soft-min" smooth objective that replaces the kinked min-triangle-area with
     a differentiable approximation, enabling efficient gradient-based optimization.
  3. Multi-start L-BFGS-B and Nelder-Mead on the full 26-D space using the soft objective
     to navigate the non-convex landscape, then a final hard-min polish.
  4. Diverse initializations: regular polygon, concentric rings, grid-like, and random.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 140.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Pre-compute all triples for fast vectorized scoring
# ---------------------------------------------------------------------------
_ti = []
_tj = []
_tk = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _ti.append(_i)
            _tj.append(_j)
            _tk.append(_k)
_ti = np.array(_ti, dtype=np.int32)
_tj = np.array(_tj, dtype=np.int32)
_tk = np.array(_tk, dtype=np.int32)


def _hull_area(pts):
    """Compute convex hull area; returns 0.0 on failure."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    x = pts[:, 0]
    y = pts[:, 1]
    cross = np.abs(
        (x[_tj] - x[_ti]) * (y[_tk] - y[_ti])
        - (y[_tj] - y[_ti]) * (x[_tk] - x[_ti])
    )
    return 0.5 * np.min(cross)


def _score_points(pts):
    """min triangle area / convex hull area (0.0 on any failure)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    mta = _min_triangle_area(pts)
    if mta < 1e-15:
        return 0.0
    return float(mta / ha)


def _soft_min_score(pts, beta=50.0):
    """
    Differentiable approximation to the min triangle area / hull area.
    Uses a soft-min (log-sum-exp) over triangle areas, normalized by hull area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    cross = np.abs(
        (x[_tj] - x[_ti]) * (y[_tk] - y[_ti])
        - (y[_tj] - y[_ti]) * (x[_tk] - x[_ti])
    )
    areas = 0.5 * cross

    # Soft-min: -1/beta * log(sum(exp(-beta * a))) approximates min(a)
    # Clamp areas to avoid overflow
    areas_clamped = np.clip(areas, 1e-15, 10.0)
    log_terms = -beta * areas_clamped
    max_log = np.max(log_terms)
    log_sum = max_log + np.log(np.sum(np.exp(log_terms - max_log)))
    soft_min = -log_sum / beta

    return float(soft_min / ha)


def _neg_soft_score(params, beta=50.0):
    """Negative soft-min score for minimization (26-D free params)."""
    return -_soft_min_score(params.reshape(N, 2), beta)


def _neg_hard_score(params):
    """Negative hard-min score for minimization (26-D free params)."""
    return -_score_points(params.reshape(N, 2))


def _make_initial_guesses(rng):
    """Generate diverse initial point configurations."""
    guesses = []

    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    a0 = _hull_area(x0)
    if a0 > 1e-12:
        x0 = x0 * np.sqrt(1.0 / a0)
    guesses.append(x0)

    # 2. Two concentric rings: 1 center + 12 outer
    center = np.array([[0.0, 0.0]])
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.0
    x0 = np.vstack([center, outer])
    a0 = _hull_area(x0)
    if a0 > 1e-12:
        x0 = x0 * np.sqrt(1.0 / a0)
    guesses.append(x0)

    # 3. Three rings: 1 center + 4 inner + 8 outer
    center = np.array([[0.0, 0.0]])
    angles_in = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    inner = np.column_stack([np.cos(angles_in), np.sin(angles_in)]) * 0.5
    angles_out = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)]) * 1.0
    x0 = np.vstack([center, inner, outer])
    a0 = _hull_area(x0)
    if a0 > 1e-12:
        x0 = x0 * np.sqrt(1.0 / a0)
    guesses.append(x0)

    # 4. Grid-like perturbation
    base = np.linspace(-1, 1, 4)
    xs = []
    ys = []
    for i in range(4):
        for j in range(4):
            xs.append(base[i] + rng.normal(0, 0.1))
            ys.append(base[j] + rng.normal(0, 0.1))
    x0 = np.column_stack([xs[:13], ys[:13]])
    a0 = _hull_area(x0)
    if a0 > 1e-12:
        x0 = x0 * np.sqrt(1.0 / a0)
    guesses.append(x0)

    # 5. Random points in unit circle
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        a0 = _hull_area(x0)
        if a0 > 1e-12:
            x0 = x0 * np.sqrt(1.0 / a0)
        guesses.append(x0)

    # 6. Regular 13-gon with slight perturbation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 += rng.normal(0, 0.05, size=(N, 2))
    a0 = _hull_area(x0)
    if a0 > 1e-12:
        x0 = x0 * np.sqrt(1.0 / a0)
    guesses.append(x0)

    return guesses


def _optimize_from_start(pts, rng, beta=50.0, max_iter_lbfgs=300, max_iter_nm=500):
    """
    Optimize from a given starting point using L-BFGS-B on soft objective,
    then Nelder-Mead polish on hard objective.
    Returns (pts, score).
    """
    best_pts = pts.copy()
    best_score = _score_points(pts)

    # Phase 1: L-BFGS-B on soft-min objective
    try:
        res = minimize(
            _neg_soft_score, pts.flatten(), args=(beta,),
            method='L-BFGS-B',
            bounds=[(-2.5, 2.5)] * (2 * N),
            options={'maxiter': max_iter_lbfgs, 'maxfun': 6000, 'ftol': 1e-12},
        )
        pts1 = res.x.reshape(N, 2)
        s1 = _score_points(pts1)
        if s1 > best_score:
            best_score = s1
            best_pts = pts1.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead on hard objective for final polish
    try:
        res = minimize(
            _neg_hard_score, best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter_nm, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True},
        )
        pts2 = res.x.reshape(N, 2)
        s2 = _score_points(pts2)
        if s2 > best_score:
            best_score = s2
            best_pts = pts2.copy()
    except Exception:
        pass

    return best_pts, best_score


def _refine_with_beta_schedule(pts, rng):
    """
    Refine using a schedule of decreasing beta values for the soft-min.
    Starts with low beta (smoother) and increases for sharper min approximation.
    """
    best_pts = pts.copy()
    best_score = _score_points(pts)

    for beta in [20.0, 40.0, 80.0, 150.0]:
        try:
            res = minimize(
                _neg_soft_score, best_pts.flatten(), args=(beta,),
                method='L-BFGS-B',
                bounds=[(-2.5, 2.5)] * (2 * N),
                options={'maxiter': 200, 'maxfun': 4000, 'ftol': 1e-13},
            )
            pts_new = res.x.reshape(N, 2)
            s = _score_points(pts_new)
            if s > best_score:
                best_score = s
                best_pts = pts_new.copy()
        except Exception:
            pass

    # Final NM polish
    try:
        res = minimize(
            _neg_hard_score, best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13,
                     'adaptive': True},
        )
        pts_new = res.x.reshape(N, 2)
        s = _score_points(pts_new)
        if s > best_score:
            best_score = s
            best_pts = pts_new.copy()
    except Exception:
        pass

    return best_pts, best_score


def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)

    best_pts = None
    best_score = -1e18

    def _update(pts):
        nonlocal best_pts, best_score
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    def _time_ok(margin):
        return (time.time() - t0) < (DEADLINE - margin)

    # Generate diverse initial guesses
    guesses = _make_initial_guesses(rng)

    # Phase 1: Optimize from each diverse start
    for guess in guesses:
        if not _time_ok(20):
            break
        pts, score = _optimize_from_start(guess, rng, beta=50.0,
                                          max_iter_lbfgs=250, max_iter_nm=400)
        _update(pts)

    # Phase 2: Beta-schedule refinement from the best found so far
    if best_pts is not None:
        if _time_ok(25):
            pts, score = _refine_with_beta_schedule(best_pts, rng)
            _update(pts)

    # Phase 3: Additional perturbed restarts from the best configuration
    if best_pts is not None:
        for pert_scale in [0.1, 0.05, 0.02]:
            if not _time_ok(15):
                break
            for _ in range(3):
                if not _time_ok(10):
                    break
                pert = best_pts.flatten() + rng.normal(0, pert_scale, size=2 * N)
                try:
                    res = minimize(
                        _neg_soft_score, pert, args=(80.0,),
                        method='L-BFGS-B',
                        bounds=[(-2.5, 2.5)] * (2 * N),
                        options={'maxiter': 200, 'maxfun': 4000, 'ftol': 1e-13},
                    )
                    pts_new = res.x.reshape(N, 2)
                    _update(pts_new)
                except Exception:
                    pass

    # Phase 4: Final hard NM polish
    if best_pts is not None:
        if _time_ok(8):
            try:
                res = minimize(
                    _neg_hard_score, best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-13,
                             'adaptive': True},
                )
                _update(res.x.reshape(N, 2))
            except Exception:
                pass

    # Fallback: regular 13-gon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x0 = np.column_stack([np.cos(angles), np.sin(angles)])
        a0 = _hull_area(x0)
        if a0 > 1e-12:
            x0 = x0 * np.sqrt(1.0 / a0)
        return x0

    return best_pts
