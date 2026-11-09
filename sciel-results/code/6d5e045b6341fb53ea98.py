"""Direct optimizer using surrogate-assisted GP with Expected Improvement for Rastrigin."""
import numpy as np
import time
import math
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted optimizer using GP + EI for Rastrigin multimodal landscape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        # Store all evaluated points for GP
        X_all = []
        y_all = []

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            X_all.append(x.copy())
            y_all.append(fx)
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        def rbf_kernel(X1, X2, ls):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-sq_dists / (2 * ls**2))

        # Phase 1: Initial diverse sampling
        # Use structured sampling to cover the space and exploit Rastrigin periodicity
        n_initial = 30
        # Mix of random and structured points
        # Rastrigin period is 2*pi ≈ 6.28, domain is [-5,5]
        # Try points at various offsets from origin and periodic shifts
        for i in range(n_initial):
            if not time_ok():
                break
            if i < 10:
                # Random uniform
                x = rng.uniform(-5, 5, dim)
            elif i < 20:
                # Points near origin with small perturbations
                x = rng.standard_normal(dim) * 0.5
            else:
                # Points at periodic shifts
                shift = 2 * math.pi
                x = rng.choice([-shift, 0, shift], size=dim) + rng.standard_normal(dim) * 0.3
                x = np.clip(x, -5, 5)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # Phase 2: Surrogate-assisted optimization with GP + EI
        # GP parameters
        length_scale = 1.5
        noise = 1e-4
        beta = 0.01  # Exploration-exploitation balance

        evals_this_run = 0
        no_improve = 0
        last_best = best_f

        # Number of candidate points to evaluate per iteration
        n_candidates_base = 100

        while time_ok() and calls_used < budget:
            # Fit GP to current data
            X_arr = np.array(X_all)
            y_arr = np.array(y_all)
            n_pts = len(X_arr)

            if n_pts < 5:
                # Not enough data, do random exploration
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
                evals_this_run += 1
                continue

            # Standardize y for numerical stability
            y_mean = y_arr.mean()
            y_scale = y_arr.std() + 1e-8
            y_std_arr = (y_arr - y_mean) / y_scale

            # Compute GP kernel matrix
            K = rbf_kernel(X_arr, X_arr, length_scale) + noise * np.eye(n_pts)

            # Generate candidate points: mixture of strategies
            n_candidates = n_candidates_base
            candidates = []

            # 40%: random uniform
            n_rand = int(n_candidates * 0.4)
            for _ in range(n_rand):
                candidates.append(rng.uniform(-5, 5, dim))

            # 30%: near best point with adaptive scale
            n_near = int(n_candidates * 0.3)
            scale_near = max(0.1, length_scale * 0.5)
            for _ in range(n_near):
                candidates.append(best_x + rng.standard_normal(dim) * scale_near)

            # 20%: periodic shifts from best
            n_periodic = int(n_candidates * 0.2)
            shift = 2 * math.pi
            for _ in range(n_periodic):
                x_base = best_x + rng.standard_normal(dim) * 0.2
                # Apply random periodic shifts
                shifts = rng.choice([-shift, 0, shift], size=dim)
                candidates.append(np.clip(x_base + shifts, -5, 5))

            # 10%: far from best for exploration
            n_far = n_candidates - n_rand - n_near - n_periodic
            for _ in range(n_far):
                # Points in opposite direction from best
                candidates.append(np.clip(-best_x + rng.standard_normal(dim) * 1.0, -5, 5))

            candidates = np.array(candidates)

            # Compute GP predictions
            K_star = rbf_kernel(candidates, X_arr, length_scale)

            # Solve for GP parameters
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n_pts))
                K_inv_y = np.linalg.solve(L.T, np.linalg.solve(L, y_std_arr))
            except np.linalg.LinAlgError:
                K_inv_y = np.linalg.lstsq(K, y_std_arr, rcond=None)[0]

            # Mean prediction
            mu_pred = K_star @ K_inv_y

            # Variance prediction
            v = np.linalg.solve(L, K_star.T)
            var_pred = 1 - np.sum(v * K_star, axis=1)
            var_pred = np.maximum(var_pred, 1e-8)

            # Expected Improvement acquisition
            f_best_std = y_std_arr.min()
            Z = (f_best_std - mu_pred) / np.sqrt(var_pred)
            EI = (f_best_std - mu_pred) * norm.cdf(Z) + np.sqrt(var_pred) * norm.pdf(Z)

            # Select top candidates by EI
            top_k = min(5, len(candidates))
            top_indices = np.argsort(EI)[::-1][:top_k]

            # Select one candidate (first one with highest EI)
            x_next = candidates[top_indices[0]]
            x_next = np.clip(x_next, -5, 5)

            # Evaluate
            fx_next = evaluate(x_next)
            evals_this_run += 1

            # Update best tracking
            if fx_next < last_best - 1e-10:
                last_best = fx_next
                no_improve = 0
            else:
                no_improve += 1

            # Adapt length scale
            if no_improve > 10:
                length_scale *= 0.7
                length_scale = max(length_scale, 0.2)
            elif no_improve == 0:
                # Improved, slightly increase exploration
                length_scale = min(length_scale * 1.1, 3.0)

            # If stuck for too long, try a larger jump
            if no_improve > 25:
                x_jump = rng.uniform(-5, 5, dim)
                evaluate(x_jump)
                evals_this_run += 1
                no_improve = 0
                length_scale = max(length_scale, 1.0)

        return (best_x, best_f)
