import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Soft-min surrogate for the minimum triangle area.
    Uses log-sum-exp for numerical stability.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)

    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    return soft_min / hull_area

def _generate_ring_points(n_out, n_in, r, theta):
    """Generate points for a two-ring configuration."""
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)

    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)

    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0

    points = _generate_ring_points(n_out, n_in, r, theta)
    return _calculate_score(points)

def _soft_score_ring_params(n_out, n_in, params, beta=50.0):
    """
    Soft-min surrogate objective for gradient-based optimization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0

    points = _generate_ring_points(n_out, n_in, r, theta)
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit, deadline):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()

    # Symmetry of the system
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g

    bounds = [(0.0, 0.999), (0.0, theta_max)]

    best_score = 0.0
    best_params = None

    # Phase 1: Differential Evolution for global search
    try:
        popsize = 10
        maxiter = 20

        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            polish=True
        )

        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass

    # Phase 2: Nelder-Mead local polish from multiple starts
    if time.time() < deadline - 5:
        starts = []
        if best_params is not None:
            starts.append(best_params)

        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max / 2, theta_max / 3, 2 * theta_max / 3]:
                starts.append([r_start, theta_start])

        for start in starts:
            if time.time() > deadline - 5:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_ring_params(n_out, n_in, p),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
            except Exception:
                pass

    # Phase 3: High-precision polish with soft-min surrogate
    if time.time() < deadline - 5 and best_params is not None:
        try:
            betas = [50.0, 100.0, 200.0, 500.0]
            current_params = best_params.copy()

            for beta in betas:
                if time.time() > deadline - 5:
                    break
                res_soft = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, beta),
                    current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-14, 'gtol': 1e-12, 'maxiter': 2000}
                )
                if -res_soft.fun > best_score:
                    best_score = -res_soft.fun
                    best_params = res_soft.x
                current_params = res_soft.x
        except Exception:
            pass

    # Phase 4: Final exact evaluation and tiny local search
    if time.time() < deadline - 3 and best_params is not None:
        try:
            # Do a fine grid search around the best params
            r_best, theta_best = best_params
            dr = 0.01
            dtheta = theta_max / 100

            for dr_i in np.linspace(-dr, dr, 5):
                for dth_i in np.linspace(-dtheta, dtheta, 5):
                    r_try = np.clip(r_best + dr_i, 0.0, 0.999)
                    th_try = (theta_best + dth_i) % theta_max

                    score_try = _score_ring_params(n_out, n_in, [r_try, th_try])
                    if score_try > best_score:
                        best_score = score_try
                        best_params = [r_try, th_try]
        except Exception:
            pass

    if best_params is None:
        return 0.0, None

    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Candidate configurations (n_out, n_in)
    # We assume outer points are on the hull.
    # n_out + n_in = 14

    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10),
        (11, 3),
        (3, 11),
        (12, 2),
        (2, 12),
        (13, 1),
        (1, 13)
    ]

    best_overall_score = 0.0
    best_overall_points = None

    # Sort candidates by likelihood (7-7 first, then others)
    for n_out, n_in in candidates:
        if time.time() > deadline - 10:  # Reserve 10s for final assembly
            break

        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 50.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 10.0
        else:
            t_limit = 5.0

        score, params = _optimize_ring(n_out, n_in, t_limit, deadline)

        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                best_overall_points = _generate_ring_points(n_out, n_in, r, theta)

    # Fallback if no valid points found
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _generate_ring_points(n_out, n_in, r, theta)

    return best_overall_points
