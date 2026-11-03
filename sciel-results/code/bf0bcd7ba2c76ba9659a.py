"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _min_triangle_ratio(points):
    """
    Compute min triangle area / convex hull area for N points.
    Higher is better. Returns 0.0 on failure.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return 0.0
    # Check for degenerate points
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area / hull_area


def _neg_ratio(points):
    return -_min_triangle_ratio(points)


def _symmetric_params_to_points(params):
    """
    Convert 14 parameters (7 angles, 7 radii) to 14 centrally-symmetric points.
    Points 0..6 are defined by (angle_i, radius_i); points 7..13 are their negatives.
    Angles are in [0, pi) and must be strictly increasing for a valid convex ordering.
    """
    angles = params[:7]
    radii = params[7:14]
    # Ensure angles are in (0, pi) and strictly increasing via soft mapping
    # Use a sigmoid-like mapping to enforce ordering
    # Actually, let's just use the raw angles but clip them
    angles = np.clip(angles, 0.01, np.pi - 0.01)
    radii = np.clip(radii, 0.05, 3.0)
    # Sort by angle to ensure convex ordering
    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]
    pts = np.zeros((14, 2))
    for i in range(7):
        pts[i] = [radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i])]
        pts[i + 7] = -pts[i]
    return pts


def _points_to_symmetric_params(points):
    """
    Convert 14 centrally-symmetric points to 14 parameters (7 angles, 7 radii).
    Assumes points are already centrally symmetric and ordered.
    """
    points = np.asarray(points, dtype=np.float64)
    # Take first 7 points as the "positive" half
    # We need to identify which 7 form one half of the symmetry
    # For now, assume points[0:7] and points[7:14] are negatives
    # But we need to ensure this is the case. Let's just use points[0:7]
    half = points[:7]
    angles = np.arctan2(half[:, 1], half[:, 0])
    angles = np.mod(angles, np.pi)  # Map to [0, pi)
    radii = np.sqrt(half[:, 0]**2 + half[:, 1]**2)
    return np.concatenate([angles, radii])


def construct():
    start_time = time.time()
    best_points = None
    best_score = 0.0

    # Initial guess: regular heptagon + its negative (centrally symmetric)
    base_angles = np.linspace(0, np.pi, 7, endpoint=False)
    base_radii = np.ones(7)
    x0_params = np.concatenate([base_angles, base_radii])
    x0_points = _symmetric_params_to_points(x0_params)
    best_points = x0_points.copy()
    best_score = _min_triangle_ratio(best_points)

    # Differential evolution bounds
    # Angles in [0.05, pi-0.05], radii in [0.1, 2.5]
    bounds = [(0.05, np.pi - 0.05)] * 7 + [(0.1, 2.5)] * 7

    # Run differential evolution with time limit
    def objective_with_time(params):
        if time.time() - start_time > _TIME_LIMIT:
            raise TimeoutError()
        pts = _symmetric_params_to_points(params)
        return _neg_ratio(pts)

    try:
        de_result = differential_evolution(
            objective_with_time,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.2),
            recombination=0.8,
            polish=True,
            init='latinhypercube',
            update_interval=1,
            callback=None,
            disp=False,
            x0=None,
            workers=1,
        )
        if de_result.x is not None and not np.any(np.isnan(de_result.x)):
            de_points = _symmetric_params_to_points(de_result.x)
            de_score = _min_triangle_ratio(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
    except (TimeoutError, Exception):
        pass

    # Check time
    if time.time() - start_time > _TIME_LIMIT - 5:
        # Not enough time for local refinement
        # Normalize to unit area
        try:
            hull = ConvexHull(best_points)
            hull_area = hull.volume
            if hull_area > 1e-12:
                best_points = best_points / np.sqrt(hull_area)
        except Exception:
            pass
        return best_points

    # Local refinement with Nelder-Mead on the best point found
    # We'll refine in the original 28D space for more flexibility
    def objective_28d(points_flat):
        if time.time() - start_time > _TIME_LIMIT:
            raise TimeoutError()
        pts = points_flat.reshape(14, 2)
        return _neg_ratio(pts)

    try:
        x0_flat = best_points.flatten()
        nm_result = minimize(
            objective_28d,
            x0_flat,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
        )
        if nm_result.x is not None and not np.any(np.isnan(nm_result.x)):
            nm_points = nm_result.x.reshape(14, 2)
            nm_score = _min_triangle_ratio(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points
    except (TimeoutError, Exception):
        pass

    # Final: normalize to unit area
    try:
        hull = ConvexHull(best_points)
        hull_area = hull.volume
        if hull_area > 1e-12:
            best_points = best_points / np.sqrt(hull_area)
    except Exception:
        pass

    return best_points
# EVOLVE-BLOCK-END
