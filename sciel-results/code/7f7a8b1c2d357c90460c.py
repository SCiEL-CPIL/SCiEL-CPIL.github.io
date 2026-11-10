# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin

    def triangle_areas(points):
        """Compute all triangle areas for a set of points."""
        n_pts = len(points)
        areas = np.zeros((n_pts * (n_pts - 1) * (n_pts - 2)) // 6)
        idx = 0
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0
                    areas[idx] = area
                    idx += 1
        return areas

    def min_triangle_area(points):
        """Compute the minimum triangle area."""
        return np.min(triangle_areas(points))

    def hull_area(points):
        """Compute the convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0.0

    def score(points):
        """Compute A_min / A_hull ratio."""
        a_min = min_triangle_area(points)
        a_hull = hull_area(points)
        if a_hull <= 0:
            return 0.0
        return a_min / a_hull

    def smooth_min_area(points, beta=100.0):
        """
        Smooth approximation of the minimum triangle area using log-sum-exp.
        This provides a differentiable surrogate for the non-smooth min operation.
        """
        areas = triangle_areas(points)
        # Use log-sum-exp for smooth minimum: -log(sum(exp(-beta * x))) / beta
        # This approximates min(x) as beta -> infinity
        neg_beta_areas = -beta * areas
        # For numerical stability, subtract max
        max_val = np.max(neg_beta_areas)
        log_sum = np.log(np.sum(np.exp(neg_beta_areas - max_val))) + max_val
        return -log_sum / beta

    def objective(params, beta=50.0):
        """
        Objective function to minimize. We want to maximize A_min/A_hull,
        so we minimize -A_min/A_hull. Using smooth surrogate for A_min.
        """
        points = params.reshape((n, 2))
        a_min_smooth = smooth_min_area(points, beta)
        a_hull = hull_area(points)
        if a_hull <= 0:
            return 1e10
        return -a_min_smooth / a_hull

    def objective_grad(params, beta=50.0):
        """
        Compute gradient of the objective using finite differences.
        """
        points = params.reshape((n, 2))
        a_min_smooth = smooth_min_area(points, beta)
        a_hull = hull_area(points)
        if a_hull <= 0:
            return np.zeros_like(params)
        
        # Numerical gradient
        grad = np.zeros_like(params)
        eps = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            obj_plus = objective(params_plus, beta)
            grad[i] = (obj_plus - objective(params, beta)) / eps
        return grad

    # Multi-start optimization
    best_points = None
    best_score = -1.0
    n_starts = 50

    rng = np.random.default_rng(seed=42)

    for start_idx in range(n_starts):
        if time.time() > deadline - 60.0:
            break

        # Generate initial points - mix of random and structured
        if start_idx % 3 == 0:
            # Random points in unit square
            points = rng.random((n, 2))
        elif start_idx % 3 == 1:
            # Grid-like initial points
            grid_x = np.linspace(0.1, 0.9, 4)
            grid_y = np.linspace(0.1, 0.9, 4)
            pts = []
            for gx in grid_x:
                for gy in grid_y:
                    pts.append([gx, gy])
            # Take first 13
            points = np.array(pts[:13])
            # Add small random perturbation
            points += rng.random((n, 2)) * 0.05
        else:
            # Concentric rings
            points = np.zeros((n, 2))
            # Center
            points[0] = [0.5, 0.5]
            # Ring 1: 4 points
            for i in range(1, 5):
                angle = 2 * np.pi * i / 4
                points[i] = [0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)]
            # Ring 2: 8 points
            for i in range(5, 13):
                angle = 2 * np.pi * (i - 4) / 8
                points[i] = [0.5 + 0.45 * np.cos(angle), 0.5 + 0.45 * np.sin(angle)]

        params = points.flatten()
        
        # Optimize
        try:
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except:
            continue

        opt_points = result.x.reshape((n, 2))
        opt_score = score(opt_points)

        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()

    # Final polish: use higher beta for more accurate min approximation
    if best_points is not None:
        params = best_points.flatten()
        try:
            result = minimize(
                lambda p: objective(p, beta=200.0),
                params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            opt_points = result.x.reshape((n, 2))
            opt_score = score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except:
            pass

    if best_points is None:
        # Fallback to random
        best_points = rng.random((n, 2))

    return best_points


# EVOLVE-BLOCK-END
