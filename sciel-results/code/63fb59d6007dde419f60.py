"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # safety margin before 150s kill


def _min_triangle_area(pts):
    """Minimum area over all C(n,3) triples."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx_j = xj - xi
            dy_j = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                a = 0.5 * abs(dx_j * (yk - yi) - (xk - xi) * dy_j)
                if a < min_a:
                    min_a = a
    return min_a


def _hull_area(pts):
    """Area of convex hull."""
    try:
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


def _symmetric_points(half):
    """
    Build 14 points from 7 half-points using central symmetry:
    points = [half[0], half[1], half[2], half[3], half[4], half[5], half[6],
              -half[0], -half[1], -half[2], -half[3], -half[4], -half[5], -half[6]]
    """
    return np.vstack([half, -half])


def _objective_sym(half):
    """Objective to minimize (negative of score)."""
    pts = _symmetric_points(half)
    return -_score(pts)


def _refine(half, time_left):
    """Local refinement using Powell and Nelder-Mead."""
    best_half = half.copy()
    best_val = _objective_sym(half)

    # Powell
    try:
        res = minimize(_objective_sym, half, method='Powell',
                       options={'maxiter': int(max(1, time_left * 5)),
                                'ftol': 1e-12, 'xtol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_half = res.x
    except Exception:
        pass

    # Nelder-Mead
    try:
        res = minimize(_objective_sym, best_half, method='Nelder-Mead',
                       options={'maxiter': int(max(1, time_left * 10)),
                                'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_half = res.x
    except Exception:
        pass

    return best_half


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # Initial guess: regular 14-gon on unit circle, take first 7 as half
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Use a slightly irregular starting point for DE
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    half_init = base[:7].copy()

    # Add small perturbation
    np.random.seed(42)
    half_init = half_init + np.random.normal(0, 0.1, half_init.shape)

    # Differential Evolution in 14D (7 points × 2 coords)
    bounds = [(-2.0, 2.0)] * 14

    best_half = half_init.copy()
    best_score = _score(_symmetric_points(half_init))

    try:
        res = differential_evolution(
            _objective_sym, bounds,
            seed=42,
            maxiter=int(max(1, (deadline - time.time()) * 2)),
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            init='sobol',
            polish=False,  # we do our own refinement
            callback=lambda x, convergence: None,
            update_interval=1,
            disp=False
        )
        if res.fun < -best_score:
            best_score = -res.fun
            best_half = res.x
    except Exception:
        pass

    # Refine
    remaining = deadline - time.time()
    if remaining > 2:
        best_half = _refine(best_half, remaining)
        best_score = -_objective_sym(best_half)

    # Final: try a few random restarts if time permits
    remaining = deadline - time.time()
    if remaining > 10:
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(100 + attempt)
            half_try = rng.uniform(-1.5, 1.5, (7, 2))
            s = _score(_symmetric_points(half_try))
            if s > best_score:
                best_score = s
                best_half = half_try
            # Quick DE on this start
            remaining = deadline - time.time()
            if remaining > 5:
                try:
                    res2 = differential_evolution(
                        _objective_sym, bounds,
                        seed=200 + attempt,
                        maxiter=int(max(1, remaining * 0.5)),
                        tol=1e-6,
                        popsize=10,
                        polish=False,
                        disp=False
                    )
                    if -res2.fun > best_score:
                        best_score = -res2.fun
                        best_half = res2.x
                except Exception:
                    pass

    # Final refinement
    remaining = deadline - time.time()
    if remaining > 1:
        best_half = _refine(best_half, remaining)

    pts = _symmetric_points(best_half)
    return pts
# EVOLVE-BLOCK-END
