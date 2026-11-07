"""GP-guided CMA-ES with adaptive restarts for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Gaussian Process surrogate-guided CMA-ES for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Track all evaluated points for GP surrogate
        all_points = []
        all_values = []

        def record_point(x, f):
            all_points.append(x.copy())
            all_values.append(f)

        # Phase 1: Initial diverse sampling using Halton sequence
        n_initial = 25

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, n_initial + 1):
            if eval_count >= budget - 150:
                break
            if time.time() > deadline:
                break
            pt = np.array([halton(i, base) * 9.0 - 4.5 for base in [2, 3, 5, 7, 11]])
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
            record_point(pt, f_val)

        # GP surrogate using RBF kernel
        class GPSurrogate:
            def __init__(self, dim, length_scale=1.0, sigma_f=1.0, sigma_n=0.01):
                self.dim = dim
                self.length_scale = length_scale
                self.sigma_f = sigma_f
                self.sigma_n = sigma_n
                self.X = None
                self.y = None
                self.K_inv = None
                self.K_inv_y = None

            def _rbf_kernel(self, X1, X2):
                # RBF kernel: exp(-0.5 * ||x1 - x2||^2 / l^2)
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=-1)
                return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

            def fit(self, X, y):
                self.X = np.asarray(X)
                self.y = np.asarray(y)
                n = len(y)
                K = self._rbf_kernel(self.X, self.X)
                K += self.sigma_n ** 2 * np.eye(n)
                # Regularize for numerical stability
                K += 1e-8 * np.eye(n)
                try:
                    self.K_inv = np.linalg.inv(K)
                    self.K_inv_y = self.K_inv @ self.y
                except np.linalg.LinAlgError:
                    self.K_inv = np.linalg.pinv(K)
                    self.K_inv_y = self.K_inv @ self.y

            def predict(self, X_new):
                X_new = np.asarray(X_new)
                if len(X_new.shape) == 1:
                    X_new = X_new[np.newaxis, :]
                K_s = self._rbf_kernel(X_new, self.X)
                K_ss = self._rbf_kernel(X_new, X_new)
                # Mean prediction
                mu = K_s @ self.K_inv_y
                # Variance prediction
                var = np.diag(K_ss) - np.sum(K_s @ self.K_inv * K_s.T, axis=1)
                var = np.maximum(var, 1e-10)
                return mu, var

        def run_cma_es(start_mean, start_sigma, max_iters, stagnation_limit=10):
            nonlocal best_x, best_f, eval_count

            pop_size = 12
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            stagnation_count = 0
            local_best_f = best_f

            for iteration in range(max_iters):
                if eval_count >= budget - 50:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                fitness = []

                for i in range(pop_size):
                    if eval_count >= budget - 50:
                        fitness.append(float('inf'))
                        population.append(mean.copy())
                        continue
                    if time.time() > deadline:
                        fitness.append(float('inf'))
                        population.append(mean.copy())
                        continue

                    z = rng.standard_normal(dim)
                    x = mean + sigma * (C_sqrt_inv @ z)
                    x = np.clip(x, -5.0, 5.0)
                    f_val = wrapped_func(x)
                    population.append(x)
                    fitness.append(f_val)
                    update_best(x, f_val)
                    record_point(x, f_val)

                # Sort by fitness
                order = np.argsort(fitness)
                population = [population[i] for i in order]
                fitness = [fitness[i] for i in order]

                # Check for stagnation
                if fitness[0] >= local_best_f * (1 + 1e-10) and local_best_f != float('inf'):
                    stagnation_count += 1
                else:
                    local_best_f = fitness[0]
                    stagnation_count = 0

                # Update mean
                old_mean = mean.copy()
                mean = sum(weights[i] * population[i] for i in range(mu))

                # Update evolution paths
                if sigma > 1e-10:
                    z_mean = C_sqrt_inv @ ((mean - old_mean) / sigma)
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                delta_h = (1 + c_c) * np.linalg.norm(p_c) / np.sqrt(mu_eff) <= 1.4 + 2 / (dim + 1)
                if delta_h:
                    rank_one = np.outer(p_c, p_c)
                    rank_mu = sum(weights[i] * np.outer(population[i] - old_mean, population[i] - old_mean) / sigma ** 2 for i in range(mu))
                    C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_h) * (np.linalg.norm(p_c) ** 2 / mu_eff - 1) * C) + c_mu * rank_mu

                # Regularize C and compute C_sqrt_inv
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                C_sqrt_inv = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

                # Restart if stagnated
                if stagnation_count >= stagnation_limit:
                    break

            return mean, sigma, C, C_sqrt_inv, p_sigma, p_c, stagnation_count

        # Phase 2: GP-guided search with CMA-ES
        # Build initial GP
        if len(all_points) >= 5:
            gp = GPSurrogate(dim, length_scale=1.5, sigma_f=1.0, sigma_n=0.01)
            gp.fit(np.array(all_points), np.array(all_values))
        else:
            gp = None

        # Main loop: alternate between GP-guided exploration and CMA-ES exploitation
        n_cycles = 5
        for cycle in range(n_cycles):
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break

            # GP-guided exploration: find point with best LCB (Lower Confidence Bound)
            if gp is not None and len(all_points) >= 5:
                # Sample candidate points
                n_candidates = 100
                candidates = rng.uniform(-4.5, 4.5, (n_candidates, dim))
                mu_pred, var_pred = gp.predict(candidates)
                # LCB: minimize mu - beta * sqrt(var)
                beta = 1.0
                lcb = mu_pred - beta * np.sqrt(var_pred)
                best_idx = np.argmin(lcb)
                next_point = candidates[best_idx]
            else:
                # No GP yet, just random
                next_point = rng.uniform(-4.5, 4.5, dim)

            f_val = wrapped_func(next_point)
            update_best(next_point, f_val)
            record_point(next_point, f_val)

            # Update GP with new point
            if len(all_points) >= 5:
                gp = GPSurrogate(dim, length_scale=1.5, sigma_f=1.0, sigma_n=0.01)
                gp.fit(np.array(all_points), np.array(all_values))

            # Run CMA-ES from current best
            if eval_count < budget - 50 and time.time() < deadline:
                mean, sigma, C, C_sqrt_inv, p_sigma, p_c, stag = run_cma_es(
                    best_x.copy(), max(0.5, sigma if 'sigma' in dir() else 1.0), 15, 8
                )

        # Phase 3: L-BFGS-B refinement from multiple starts
        start_points = [best_x.copy()]

        # Add local perturbations at different scales
        for scale in [0.3, 0.15, 0.05]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.05:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Run L-BFGS-B from each starting point
        for sp in unique_starts:
            if eval_count >= budget - 10:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 20, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-7}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Use remaining budget for local random search
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
