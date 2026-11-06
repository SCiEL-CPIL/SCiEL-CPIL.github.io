import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n = 8
    
    # Initial guess: Regular Octagon
    theta = 2 * np.pi * np.arange(n) / n
    points_init = np.column_stack([np.cos(theta), np.sin(theta)])

    def log_score(pts):
        """Compute log of the product of all pairwise distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        # Avoid log(0)
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(n, k=1)
        log_delta = np.sum(np.log(dists[iu]))
        # The score is Delta / n^n. Delta is product over ORDERED pairs.
        # Product over ordered pairs = (Product over unordered pairs)^2
        # So log(Delta) = 2 * log(Product over unordered pairs)
        # log(Score) = 2 * log_delta - n * log(n)
        return 2.0 * log_delta - n * np.log(n)

    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    def objective(x, penalty_weight=100.0):
        pts = x.reshape(n, 2)
        ls = log_score(pts)
        d = diameter(pts)
        penalty = 0.0
        if d > 2.0:
            penalty = penalty_weight * (d - 2.0) ** 2
        # We want to maximize ls, so minimize -ls + penalty
        return -ls + penalty

    def gradient(x, penalty_weight=100.0):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = objective(x, penalty_weight)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective(x_plus, penalty_weight) - f0) / eps
        return grad

    best_score = -np.inf
    best_points = points_init.copy()

    # Evaluate initial guess
    current_score = log_score(points_init)
    if current_score > best_score:
        best_score = current_score
        best_points = points_init.copy()

    # Strategy: Use Differential Evolution for global search
    # Bounds: coordinates between -2 and 2 (since diameter is 2, points shouldn't be too far)
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # Run DE for a short time to find good basins
    if time.time() < deadline - 60:
        try:
            # DE is slow, so limit iterations
            de_result = differential_evolution(
                lambda x: objective(x, 100.0),
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=False, # We will polish manually
                update_interval=10,
                callback=lambda xk, convergence: time.time() < deadline - 60
            )
            
            if de_result.success or de_result.fun < objective(points_init.flatten(), 100.0):
                de_pts = de_result.x.reshape(n, 2)
                # Polish with L-BFGS-B
                res = minimize(
                    lambda x: objective(x, 1000.0),
                    de_result.x,
                    jac=lambda x: gradient(x, 1000.0),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.success:
                    opt_pts = res.x.reshape(n, 2)
                    opt_score = log_score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
        except Exception as e:
            pass

    # Strategy: Random restarts with local optimization
    # Try a few random starting points
    rng = np.random.RandomState(123)
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random points in a square
        random_pts = rng.uniform(-1.5, 1.5, (n, 2))
        x0 = random_pts.flatten()
        
        try:
            res = minimize(
                lambda x: objective(x, 100.0),
                x0,
                jac=lambda x: gradient(x, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                opt_pts = res.x.reshape(n, 2)
                # Scale to diameter 2
                d = diameter(opt_pts)
                if d > 0:
                    opt_pts = opt_pts * (2.0 / d)
                opt_score = log_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
        except Exception:
            continue

    # Final Polish on the best point found so far
    if time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda x: objective(x, 10000.0), # High penalty to enforce constraint strictly
                x0,
                jac=lambda x: gradient(x, 10000.0),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success:
                opt_pts = res.x.reshape(n, 2)
                d = diameter(opt_pts)
                if d > 0:
                    opt_pts = opt_pts * (2.0 / d)
                opt_score = log_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
        except Exception:
            pass

    # Ensure diameter is exactly 2
    d = diameter(best_points)
    if d > 0:
        best_points = best_points * (2.0 / d)

    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in best_points]
