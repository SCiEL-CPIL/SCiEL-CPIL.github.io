# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds

    n = 16
    d = 2

    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, beta=50.0):
        """
        Smooth objective using log-sum-exp approximations for min and max.
        Minimizes -log(R_smooth) where R_smooth uses smooth min/max.
        """
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()

        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 1e10

        # Avoid zero/negative distances
        dists = np.maximum(dists, 1e-10)

        # Smooth max: (1/beta) * log(sum(exp(beta * d)))
        # Smooth min: -(1/beta) * log(sum(exp(-beta * d)))
        # R_smooth = d_min_smooth / d_max_smooth
        # We want to maximize R, so minimize -log(R) = log(d_max) - log(d_min)

        # For numerical stability, use log-sum-exp trick
        max_d = np.max(dists)
        min_d = np.min(dists)

        # Smooth log(max): log(max) ≈ max_d + (1/beta)*log(sum(exp(beta*(d - max_d))))
        # Smooth log(min): log(min) ≈ min_d - (1/beta)*log(sum(exp(-beta*(d - min_d))))

        exp_pos = np.exp(beta * (dists - max_d))
        log_max_smooth = max_d + (1.0 / beta) * np.log(np.sum(exp_pos))

        exp_neg = np.exp(-beta * (dists - min_d))
        log_min_smooth = min_d - (1.0 / beta) * np.log(np.sum(exp_neg))

        # Objective: minimize log(d_max_smooth) - log(d_min_smooth)
        obj = log_max_smooth - log_min_smooth
        return obj

    def exact_objective(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()

        points = params.reshape(n, d)
        R = compute_R(points)
        return -R  # minimize -R

    best_params = None
    best_score = -1e10

    # Multiple random restarts with different initial configurations
    rng = np.random.RandomState(42)

    for restart in range(8):
        if time.time() > start_time + deadline - safety_margin:
            break

        # Generate initial points: mix of grid, random, and structured patterns
        if restart == 0:
            # 4x4 grid
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            init_points = np.column_stack((X.ravel(), Y.ravel()))
        elif restart == 1:
            # Random points in unit square
            init_points = rng.uniform(-2, 2, size=(n, d))
        elif restart == 2:
            # Circular arrangement
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radii = np.ones(n) * 1.5
            init_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        elif restart == 3:
            # Two concentric circles
            angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
            r1, r2 = 0.8, 2.0
            init_points = np.column_stack([
                np.concatenate([r1 * np.cos(angles1), r2 * np.cos(angles2)]),
                np.concatenate([r1 * np.sin(angles1), r2 * np.sin(angles2)])
            ])
        elif restart == 4:
            # Hexagonal lattice pattern
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            r_inner = 0.5
            r_outer = 1.8
            inner = np.column_stack([r_inner * np.cos(angles), r_inner * np.sin(angles)])
            outer = np.column_stack([r_outer * np.cos(angles + np.pi/6), r_outer * np.sin(angles + np.pi/6)])
            # Add 4 more points
            extra_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
            r_mid = 1.2
            mid = np.column_stack([r_mid * np.cos(extra_angles), r_mid * np.sin(extra_angles)])
            init_points = np.vstack([inner, outer, mid])
        elif restart == 5:
            # Square lattice with perturbation
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            init_points = np.column_stack((X.ravel(), Y.ravel()))
            init_points += rng.normal(0, 0.1, size=(n, d))
        elif restart == 6:
            # Fibonacci-like spiral
            t = np.arange(n)
            r = 0.3 * np.sqrt(t + 1)
            theta = t * 2.39996322973  # golden angle
            init_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        else:
            # Pure random
            init_points = rng.uniform(-3, 3, size=(n, d))

        init_params = init_points.flatten()

        # L-BFGS-B optimization with smooth objective
        try:
            result = minimize(
                smooth_objective,
                init_params,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * (n * d),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 1e10:
                # Evaluate exact score
                candidate_points = result.x.reshape(n, d)
                score = compute_R(candidate_points)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass

        # Local refinement with Powell on the best found so far from this restart
        if best_params is not None:
            try:
                result_ref = minimize(
                    exact_objective,
                    best_params,
                    method='Powell',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                candidate_points = result_ref.x.reshape(n, d)
                score = compute_R(candidate_points)
                if score > best_score:
                    best_score = score
                    best_params = result_ref.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if best_params is not None:
        try:
            result_final = minimize(
                exact_objective,
                best_params,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            candidate_points = result_final.x.reshape(n, d)
            score = compute_R(candidate_points)
            if score > best_score:
                best_score = score
                best_params = result_final.x.copy()
        except Exception:
            pass

    # Convert back to points
    if best_params is not None:
        points = best_params.reshape(n, d)
    else:
        # Fallback
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

    if points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

    return points

# EVOLVE-BLOCK-END
