# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area_exact(pts):
    """Compute the exact minimum triangle area among all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    if len(pts) < 3:
        return 0.0
    a_min = _min_triangle_area_exact(pts)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _triangle_areas_vectorized(pts):
    """Compute all 286 triangle areas vectorized."""
    n = len(pts)
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    x1, y1 = pts[i_idx, 0], pts[i_idx, 1]
    x2, y2 = pts[j_idx, 0], pts[j_idx, 1]
    x3, y3 = pts[k_idx, 0], pts[k_idx, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas


def _soft_min_triangle_area(pts, beta):
    """Soft-min of triangle areas using log-sum-exp for smoothness."""
    areas = _triangle_areas_vectorized(pts)
    # Clip to avoid numerical issues
    areas = np.maximum(areas, 1e-15)
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum / beta
    return soft_min


def _hull_area_smooth(pts):
    """Compute hull area, with a small penalty if hull is degenerate."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Penalize degenerate configurations
        return 1e-10


def _objective(x, beta):
    """Objective to minimize: -A_min_soft / A_hull."""
    pts = np.column_stack([x[0:13], x[13:26]])
    soft_min = _soft_min_triangle_area(pts, beta)
    hull = _hull_area_smooth(pts)
    if hull <= 0 or not np.isfinite(soft_min) or not np.isfinite(hull):
        return 1e10
    return -soft_min / hull


def _gradient(x, beta):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros(26)
    f0 = _objective(x, beta)
    for i in range(26):
        xp = x.copy()
        xp[i] += eps
        fp = _objective(xp, beta)
        xm = x.copy()
        xm[i] -= eps
        fm = _objective(xm, beta)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def _regular_polygon(n=13):
    """Create regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 580  # Safety margin from 630s

    n = 13
    best_pts = _regular_polygon(n)
    best_score = _score(best_pts)

    # Try multiple initializations with different structures
    initializations = []

    # 1. Regular polygon
    initializations.append(_regular_polygon(n))

    # 2. Two concentric rings (5 + 8 or 6 + 7)
    for inner_n, outer_n in [(5, 8), (6, 7), (4, 9)]:
        inner_angles = np.linspace(0, 2*np.pi, inner_n, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, outer_n, endpoint=False) + np.pi/outer_n
        inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initializations.append(np.vstack([inner_pts, outer_pts]))

    # 3. Grid-like arrangement
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([0.3 + 0.4*i, 0.3 + 0.4*j])
    grid_pts = np.array(grid_pts[:n])
    initializations.append(grid_pts)

    # 4. Slightly perturbed regular polygon
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.1, (n, 2))
        initializations.append(perturbed)

    # Optimize each initialization
    beta_schedule = [50, 100, 200, 500, 1000, 2000]

    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60:
            break

        # Flatten to 26-d vector
        x0 = init_pts.flatten()

        # Anneal beta: start with low beta (smoother), increase for sharper min
        current_x = x0.copy()
        for beta in beta_schedule:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 30:
                break

            # L-BFGS-B optimization
            try:
                result = minimize(
                    lambda x: _objective(x, beta),
                    current_x,
                    method='L-BFGS-B',
                    jac=lambda x: _gradient(x, beta),
                    options={
                        'maxiter': 500,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                if result.fun < _objective(current_x, beta) - 1e-10:
                    current_x = result.x
            except Exception:
                pass

            # Nelder-Mead polish
            try:
                nm_result = minimize(
                    lambda x: _objective(x, beta),
                    current_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-12,
                        'fatol': 1e-14
                    }
                )
                if nm_result.fun < _objective(current_x, beta) - 1e-10:
                    current_x = nm_result.x
            except Exception:
                pass

        # Evaluate final result
        final_pts = np.column_stack([current_x[0:13], current_x[13:26]])
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final high-precision polish with exact objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20 and best_score > 0:
        x_final = best_pts.flatten()

        # Use exact objective for final polish
        def exact_obj(x):
            pts = np.column_stack([x[0:13], x[13:26]])
            s = _score(pts)
            return -s if s > 0 else 1e10

        # Nelder-Mead with tight tolerances
        try:
            nm_final = minimize(
                exact_obj,
                x_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-14,
                    'fatol': 1e-16
                }
            )
            if nm_final.fun < -best_score:
                best_pts = np.column_stack([nm_final.x[0:13], nm_final.x[13:26]])
                best_score = -nm_final.fun
        except Exception:
            pass

    # Validate result
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(n)

    return best_pts


# EVOLVE-BLOCK-END
