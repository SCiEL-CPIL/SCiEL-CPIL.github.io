"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# EVOLVE-BLOCK-START


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate a regular n-gon."""
    angles = angle_offset + 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def construct():
    """Construct 14 points in a convex region to maximize min triangle area / hull area."""
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill

    # Start with regular polygon as initial guess
    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    # Try multiple starting configurations and local optimizations
    # Strategy: use angular parameterization on unit circle for convex position
    # Then optimize the angles to maximize min triangle area / hull area

    def _angular_objective(angles):
        """Objective: negative of min triangle area / hull area for angles on unit circle."""
        if time.time() > deadline - 5.0:
            return 1e10  # Large penalty to stop optimization
        # Sort angles to ensure convex position
        angles = np.sort(angles)
        # Place points on unit circle at these angles
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        # For points on unit circle, hull area is the area of the polygon
        # which can be computed directly
        hull_area = _convex_hull_area(points)
        if hull_area < 1e-12:
            return 1e10
        min_tri = _min_triangle_area(points)
        return -min_tri / hull_area

    def _angular_gradient(angles):
        """Numerical gradient for angular objective."""
        if time.time() > deadline - 5.0:
            return np.zeros_like(angles)
        angles = np.sort(angles)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull_area = _convex_hull_area(points)
        if hull_area < 1e-12:
            return np.zeros_like(angles)
        min_tri = _min_triangle_area(points)
        score = min_tri / hull_area
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(angles)
        for i in range(N):
            angles_plus = angles.copy()
            angles_plus[i] += eps
            angles_plus = np.sort(angles_plus)
            points_plus = np.column_stack([np.cos(angles_plus), np.sin(angles_plus)])
            hull_area_plus = _convex_hull_area(points_plus)
            if hull_area_plus < 1e-12:
                grad[i] = 0
                continue
            min_tri_plus = _min_triangle_area(points_plus)
            score_plus = min_tri_plus / hull_area_plus
            grad[i] = (score_plus - score) / eps
        return -grad  # Negative because we're minimizing negative score

    # Try multiple random initializations of angles
    np.random.seed(42)
    n_trials = 0
    max_trials = 50

    for trial in range(max_trials):
        if time.time() > deadline - 10.0:
            break

        # Random angles
        angles0 = np.sort(np.random.uniform(0, 2 * np.pi, N))
        
        # Use Nelder-Mead for gradient-free optimization
        try:
            result = minimize(
                _angular_objective,
                angles0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.success or result.fun < 0:
                angles_opt = np.sort(result.x)
                points_opt = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
                score_opt = _score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt
        except Exception:
            pass

        n_trials += 1

    # Also try starting from regular polygon with small perturbations
    for pert_scale in [0.01, 0.05, 0.1, 0.2]:
        if time.time() > deadline - 10.0:
            break
        angles0 = np.sort(2 * np.pi * np.arange(N) / N + pert_scale * np.random.randn(N))
        angles0 = np.sort(angles0)
        try:
            result = minimize(
                _angular_objective,
                angles0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.success or result.fun < 0:
                angles_opt = np.sort(result.x)
                points_opt = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
                score_opt = _score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt
        except Exception:
            pass

    # Final polish: use BFGS with numerical gradient on the best solution
    if time.time() < deadline - 15.0:
        angles_best = np.arctan2(best_points[:, 1], best_points[:, 0])
        angles_best = np.sort(angles_best)
        try:
            result = minimize(
                _angular_objective,
                angles_best,
                method='BFGS',
                jac=_angular_gradient,
                options={
                    'maxiter': 200,
                    'gtol': 1e-8,
                    'factr': 1e10
                }
            )
            if result.fun < 0:
                angles_opt = np.sort(result.x)
                points_opt = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
                score_opt = _score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
