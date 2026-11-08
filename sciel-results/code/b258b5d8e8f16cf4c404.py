"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        if len(points) < 3:
            return 0.0
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _soft_min_triangle_area(points, beta):
    """
    Soft minimum of triangle areas using log-sum-exp (lower beta = sharper min).
    Returns a smooth approximation of the minimum triangle area.
    """
    n = len(points)
    areas = np.empty((n - 2) * (n - 1) // 2 * (1), dtype=float)  # placeholder
    # Compute all triangle areas
    idx = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    m = len(idx)
    areas = np.empty(m)
    for t, (i, j, k) in enumerate(idx):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        areas[t] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    
    # Soft min via log-sum-exp: soft_min(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # For stability, subtract max
    neg_beta_x = -beta * areas
    max_val = np.max(neg_beta_x)
    if max_val > 700:
        # overflow guard
        return np.min(areas)
    s = np.sum(np.exp(neg_beta_x - max_val))
    soft_min_val = -(1.0 / beta) * (np.log(s) + max_val)
    return soft_min_val


def _soft_score(points, beta):
    """Soft score = soft_min_triangle_area / convex_hull_area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    sm_area = _soft_min_triangle_area(points, beta)
    return sm_area / hull_area


def _neg_soft_score_flat(x, beta):
    """Objective for DE: minimize negative soft score."""
    points = x.reshape(N, 2)
    # Keep points in a reasonable box to avoid degenerate configs
    if np.any(np.abs(points) > 10):
        return 1e10
    s = _soft_score(points, beta)
    return -s


def _neg_exact_score_flat(x):
    """Objective for refinement: minimize negative exact score."""
    points = x.reshape(N, 2)
    if np.any(np.abs(points) > 10):
        return 1e10
    s = _score(points)
    return -s


def _regular_polygon(n, scale=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return scale * np.column_stack((np.cos(angles), np.sin(angles)))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    rng = np.random.default_rng(12345)

    # Initial best: regular 13-gon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    # Define bounds for each coordinate: [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * N)

    # Phase 1: Differential Evolution with soft-min objective (coarse)
    # Use a moderate beta for smoothness
    beta_coarse = 5.0
    de_result_coarse = differential_evolution(
        _neg_soft_score_flat,
        bounds,
        args=(beta_coarse,),
        seed=42,
        maxiter=80,
        popsize=15,
        tol=1e-6,
        mutation=(0.4, 1.0),
        recombination=0.7,
        polish=False,
        init='sobol'
    )
    if time.time() > deadline:
        return best_points

    cand = de_result_coarse.x.reshape(N, 2)
    cand_score = _score(cand)
    if cand_score > best_score:
        best_score = cand_score
        best_points = cand

    # Phase 2: Differential Evolution with sharper soft-min (medium)
    beta_medium = 20.0
    # Initialize DE around the best found so far
    de_result_medium = differential_evolution(
        _neg_soft_score_flat,
        bounds,
        args=(beta_medium,),
        seed=99,
        maxiter=60,
        popsize=12,
        tol=1e-7,
        mutation=(0.3, 0.8),
        recombination=0.6,
        polish=False,
        init='sobol'
    )
    if time.time() > deadline:
        return best_points

    cand = de_result_medium.x.reshape(N, 2)
    cand_score = _score(cand)
    if cand_score > best_score:
        best_score = cand_score
        best_points = cand

    # Phase 3: DE with very sharp soft-min (fine)
    beta_fine = 80.0
    de_result_fine = differential_evolution(
        _neg_soft_score_flat,
        bounds,
        args=(beta_fine,),
        seed=777,
        maxiter=40,
        popsize=10,
        tol=1e-8,
        mutation=(0.2, 0.6),
        recombination=0.5,
        polish=False,
        init='sobol'
    )
    if time.time() > deadline:
        return best_points

    cand = de_result_fine.x.reshape(N, 2)
    cand_score = _score(cand)
    if cand_score > best_score:
        best_score = cand_score
        best_points = cand

    # Phase 4: Nelder-Mead refinement on exact objective from best DE candidate
    if time.time() < deadline - 20:
        def objective_exact(x):
            pts = x.reshape(N, 2)
            if np.any(np.abs(pts) > 10):
                return 1e10
            return -_score(pts)

        # Try a few Nelder-Mead restarts from best and perturbed
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            if restart == 0:
                x0 = best_points.flatten()
            else:
                x0 = best_points.flatten() + rng.normal(0, 0.02, size=2 * N)

            res = minimize(
                objective_exact,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            cand = res.x.reshape(N, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand

    # Final check
    if time.time() > deadline:
        return best_points

    # One more DE pass with exact objective if time permits
    remaining = deadline - time.time()
    if remaining > 30:
        de_result_exact = differential_evolution(
            _neg_exact_score_flat,
            bounds,
            seed=2024,
            maxiter=30,
            popsize=8,
            tol=1e-8,
            mutation=(0.2, 0.5),
            recombination=0.4,
            polish=False,
            init='sobol'
        )
        cand = de_result_exact.x.reshape(N, 2)
        cand_score = _score(cand)
        if cand_score > best_score:
            best_score = cand_score
            best_points = cand

    return best_points
# EVOLVE-BLOCK-END
