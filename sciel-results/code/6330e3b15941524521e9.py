import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 115.0
_SAFETY = 8.0
_START = time.time()


def _cage_points(r1, r2, theta):
    """Two concentric squares: outer radius r1 at angles pi/4 + k*pi/2,
    inner radius r2 at angles theta + k*pi/2."""
    pts = np.empty((8, 2))
    for k in range(4):
        a = math.pi / 4.0 + k * math.pi / 2.0
        pts[2 * k] = [r1 * math.cos(a), r1 * math.sin(a)]
        pts[2 * k + 1] = [r2 * math.cos(theta + k * math.pi / 2.0),
                          r2 * math.sin(theta + k * math.pi / 2.0)]
    return pts


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())


def _log_product(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    return float(np.log(d).sum())


def _soft_max_diameter(pts, beta=50.0):
    """Smooth approximation of max pairwise distance using log-sum-exp."""
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 0.0)
    # Use log-sum-exp for numerical stability
    d_max = d.max()
    if d_max < 1e-12:
        return d_max
    log_sum = d_max + math.log(np.sum(np.exp(d - d_max)))
    return math.exp(log_sum / beta)


def _score_from_log_product_and_diameter(lp, dia):
    n = _N
    return math.exp(lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n))


def _score(pts):
    dia = _diameter(pts)
    if dia <= 1e-12:
        return 0.0
    return _score_from_log_product_and_diameter(_log_product(pts), dia)


def _objective(pts):
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e12
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)


def _smooth_objective(pts, beta=50.0):
    """Smooth objective using soft-max diameter for better gradient flow."""
    dia = _soft_max_diameter(pts, beta)
    if dia < 1e-8:
        return 1e12
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)


def _refine(pts):
    """
    Multi-stage local refinement.
    1. Smooth objective L-BFGS-B (soft-max diameter) for better gradient flow.
    2. Exact objective L-BFGS-B.
    3. SLSQP for different landscape navigation.
    4. Final high-precision L-BFGS-B with very tight tolerances.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)

    # Stage 1: Smooth objective (soft-max diameter) L-BFGS-B
    if _remaining() > 3.0:
        try:
            res = minimize(_smooth_objective, pts.ravel(), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-10})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 2: Exact objective L-BFGS-B
    if _remaining() > 3.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: SLSQP
    if _remaining() > 4.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-13})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 4: High-precision polish
    if _remaining() > 2.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    return best_pts, best_score


def _cage_objective(params):
    """Objective for 3-parameter cage: minimize -score."""
    r1, r2, theta = params
    if r1 < 1e-8 or r2 < 1e-8:
        return 1e10
    pts = _cage_points(r1, r2, theta)
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e10
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)


def _cage_grid_search():
    """Fine grid search over (r1, r2, theta) for the two-concentric-squares cage."""
    best_params = None
    best_obj = 1e10

    # Coarse grid - wider range
    r1_vals = np.linspace(0.7, 1.6, 35)
    r2_vals = np.linspace(0.05, 1.1, 35)
    theta_vals = np.linspace(0.0, math.pi / 2.0, 25)

    for r1 in r1_vals:
        for r2 in r2_vals:
            for theta in theta_vals:
                obj = _cage_objective([r1, r2, theta])
                if obj < best_obj:
                    best_obj = obj
                    best_params = [r1, r2, theta]

    # Local refinement in 3D
    if best_params is not None:
        try:
            res = minimize(_cage_objective, best_params, method='L-BFGS-B',
                           bounds=[(1e-6, 3.0), (1e-6, 3.0), (0.0, math.pi / 2.0)],
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.tolist()
        except Exception:
            pass

        # Try SLSQP too
        try:
            res = minimize(_cage_objective, best_params, method='SLSQP',
                           bounds=[(1e-6, 3.0), (1e-6, 3.0), (0.0, math.pi / 2.0)],
                           options={'maxiter': 800, 'ftol': 1e-14})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.tolist()
        except Exception:
            pass

    return best_params, -best_obj


def _make_cage_init(kind, rng):
    """Generate diverse initial configurations."""
    if kind == 0:  # Concentric squares
        a = 1.0
        b = 0.5
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([a * math.cos(ang), a * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([b * math.cos(ang), b * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 1:  # Nested rings (outer 5, inner 3)
        pts = []
        for k in range(5):
            ang = 2 * math.pi * k / 5
            pts.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(3):
            ang = 2 * math.pi * k / 3
            pts.append([0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 2:  # Deformed octagon
        pts = []
        for k in range(_N):
            ang = 2 * math.pi * k / _N
            r = 1.0 + 0.15 * rng.uniform(-1, 1)
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 3:  # Two circles
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([0.6 * math.cos(ang), 0.6 * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 4:  # Random
        return rng.uniform(-1.0, 1.0, size=(8, 2))
    if kind == 5:  # Concentric squares with different ratio
        a = 1.0
        b = 0.7
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([a * math.cos(ang), a * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4 + 0.2
            pts.append([b * math.cos(ang), b * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 6:  # Rectangle-based
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([1.2 * math.cos(ang), 0.8 * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 7:  # Star pattern
        pts = []
        for k in range(8):
            ang = 2 * math.pi * k / 8
            r = 1.0 if k % 2 == 0 else 0.6
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        return np.array(pts, dtype=float)
    # kind 8+: Random
    return rng.uniform(-1.0, 1.0, size=(8, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # 1. Seed with regular octagon and refine
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    best, best_score = _refine(oct_pts)

    # 2. Focused 3D cage search (two concentric squares)
    if _remaining() > 20.0:
        cage_params, cage_score = _cage_grid_search()
        if cage_params is not None:
            cage_pts = _cage_points(*cage_params)
            cand, s = _refine(cage_pts)
            if s > best_score:
                best, best_score = cand, s

    # 3. Refine diverse combinatorial topologies
    for kind in range(1, 8):
        if _remaining() < 5.0:
            break
        init = _make_cage_init(kind, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # 4. Differential Evolution for global search
    if _remaining() > 30.0:
        def de_objective(x):
            return _objective(x.reshape(_N, 2))

        bounds = [(-2.0, 2.0)] * (2 * _N)
        try:
            de_res = differential_evolution(
                de_objective, bounds, seed=42, maxiter=100, popsize=20,
                tol=1e-8, polish=False, init='sobol'
            )
            cand = de_res.x.reshape(_N, 2)
            cand, s = _refine(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 5. More DE runs with different seeds if time permits
    if _remaining() > 30.0:
        for seed in [123, 456]:
            if _remaining() < 30.0:
                break
            try:
                de_res = differential_evolution(
                    de_objective, bounds, seed=seed, maxiter=80, popsize=15,
                    tol=1e-8, polish=False, init='latinhypercube'
                )
                cand = de_res.x.reshape(_N, 2)
                cand, s = _refine(cand)
                if s > best_score:
                    best, best_score = cand, s
            except Exception:
                pass

    # 6. Extra multi-start random + local refinement while budget remains
    while _remaining() > 8.0:
        init = _make_cage_init(8, rng)  # random
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # Final high-precision refinement of the best configuration
    if _remaining() > 3.0:
        cand, s = _refine(best)
        if s > best_score:
            best, best_score = cand, s

    best = np.asarray(best, dtype=float).reshape(_N, 2)

    # Ensure distinct points and bounded coordinates
    for i in range(_N):
        for j in range(i + 1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]

    # Clamp coordinates
    best = np.clip(best, -1e6, 1e6)

    return [[float(x), float(y)] for (x, y) in best]
