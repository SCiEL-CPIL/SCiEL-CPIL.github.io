"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for initial exploration.
    - Phase 2: Gaussian Process surrogate to guide exploration via EI.
    - Phase 3: Multi-start L-BFGS-B local refinement from top candidates.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        n_initial = min(64, self.budget // 5)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 50:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        evaluations.sort(key=lambda t: t[0])

        # Phase 2: Gaussian Process surrogate to guide exploration
        # Build a simple GP model from initial evaluations
        X_obs = np.array([x for _, x in evaluations])
        y_obs = np.array([fx for fx, _ in evaluations])

        # Normalize y for GP stability
        y_min = y_obs.min()
        y_max = y_obs.max()
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_norm = (y_obs - y_min) / y_range

        # Simple GP with RBF kernel
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            # Compute pairwise distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))

        def gp_predict(X_new, X_obs, y_obs, length_scale=1.0, variance=1.0, noise=1e-8):
            K = rbf_kernel(X_obs, X_obs, length_scale, variance)
            K += noise * np.eye(len(X_obs))
            K_star = rbf_kernel(X_obs, X_new, length_scale, variance)
            K_new = rbf_kernel(X_new, X_new, length_scale, variance)

            try:
                K_inv_y = np.linalg.solve(K, y_obs)
            except np.linalg.LinAlgError:
                K_inv_y = np.linalg.lstsq(K, y_obs, rcond=None)[0]

            mu = K_star.T @ K_inv_y
            v = np.linalg.solve(K, K_star)
            var = np.diag(K_new - v.T @ v)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        # Adaptively choose length scale based on data spread
        if len(X_obs) >= 2:
            dists = np.array([np.linalg.norm(X_obs[i] - X_obs[j]) 
                             for i in range(len(X_obs)) 
                             for j in range(i+1, len(X_obs))])
            length_scale = np.median(dists) if len(dists) > 0 else 1.0
            length_scale = max(0.1, min(5.0, length_scale))
        else:
            length_scale = 1.0

        # Variance scale based on function value spread
        variance = max(0.1, y_range ** 2) if y_range > 0 else 1.0

        # Phase 2a: Use EI to select next points
        n_gp_steps = min(30, self.budget - calls_used - 30)
        gp_candidates = []

        for _ in range(n_gp_steps):
            if calls_used >= self.budget - 30:
                break
            if time.time() > deadline - 10:
                break

            # Generate candidate points
            n_candidates = 50
            X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))

            # Predict mean and variance at candidates
            mu, sigma = gp_predict(X_cand, X_obs, y_norm, length_scale, variance)

            # Expected Improvement
            current_best = y_norm.min()
            Z = (current_best - mu) / (sigma + 1e-10)
            from scipy.stats import norm
            EI = (current_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Select point with highest EI
            idx = np.argmax(EI)
            x_next = X_cand[idx]

            # Evaluate
            fx = safe_call(x_next)
            evaluations.append((fx, x_next))
            evaluations.sort(key=lambda t: t[0])

            # Update GP training data
            X_obs = np.array([x for _, x in evaluations[:min(100, len(evaluations))]])
            y_obs = np.array([fx for fx, _ in evaluations[:min(100, len(evaluations))]])
            y_min = y_obs.min()
            y_max = y_obs.max()
            y_range = y_max - y_min if y_max > y_min else 1.0
            y_norm = (y_obs - y_min) / y_range

        # Phase 3: Local refinement from the best few points
        remaining_budget = self.budget - calls_used

        # Select top candidates for local refinement
        n_restarts = min(5, len(evaluations))
        max_iter_per_restart = max(10, remaining_budget // (n_restarts + 1))

        bounds = [(-5, 5)] * self.dim

        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_used >= self.budget - 5:
                break

            fx0, x0 = evaluations[i]

            def lbfgs_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    lbfgs_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_per_restart,
                        'gtol': 1e-10,
                        'ftol': 1e-12,
                    }
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try a few more random restarts with local search
        if calls_used < self.budget - 20 and time.time() < deadline - 30:
            possible_restarts = (self.budget - calls_used) // 30
            for _ in range(possible_restarts):
                if time.time() > deadline - 10:
                    break
                if calls_used >= self.budget - 10:
                    break

                x0 = rng.uniform(-5, 5, self.dim)
                fx0 = safe_call(x0)

                if calls_used < self.budget - 5:
                    def lbfgs_func2(x):
                        return safe_call(x)
                    try:
                        res = minimize(
                            lbfgs_func2,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 20,
                                'gtol': 1e-8,
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
