# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective(points_flat):
    """Objective function to minimize (negative log of score for numerical stability)."""
    n = 16
    points = points_flat.reshape(n, 2)
    score = _compute_score(points)
    if score <= 0:
        return 1e10
    return -np.log(score)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    # Multiple initial configurations
    initial_configs = []

    # 1. 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    initial_configs.append(grid)

    # 2. Two concentric squares (4 points on inner square, 12 on outer)
    # Inner square: 4 points
    inner = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    # Outer: 12 points arranged in 3 squares of 4? Or a 4x4 minus corners?
    # Let's do a 4x4 grid scaled and then optimize
    grid2 = np.array([[i-1.5, j-1.5] for i in range(4) for j in range(4)], dtype=float)
    initial_configs.append(grid2 * 2.0)

    # 3. Random configurations
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.standard_normal((n, d))
        initial_configs.append(pts)

    # 4. Hexagonal-ish packing
    # 4 rows of 4 points with hexagonal offset
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts, dtype=float)
    initial_configs.append(hex_pts)

    best_points = initial_configs[0].copy()
    best_score = _compute_score(best_points)

    # Try to optimize each configuration
    from scipy.optimize import minimize

    for config in initial_configs:
        if time.time() > deadline:
            break

        flat = config.flatten()
        try:
            result = minimize(
                _objective,
                flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success and time.time() < deadline:
                opt_points = result.x.reshape(n, 2)
                opt_score = _compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            continue

    # If we still have time, do more random restarts
    if time.time() < deadline - 60:
        for _ in range(20):
            if time.time() > deadline:
                break
            rng = np.random.default_rng()
            pts = rng.standard_normal((n, d))
            flat = pts.flatten()
            try:
                result = minimize(
                    _objective,
                    flat,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success and time.time() < deadline:
                    opt_points = result.x.reshape(n, 2)
                    opt_score = _compute_score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
            except Exception:
                continue

    # Ensure the result is valid
    if best_points.shape != (n, d):
        best_points = initial_configs[0]

    return best_points


# EVOLVE-BLOCK-END
