"""CMA-ES with multi-start and adaptive restarts for Rastrigin rotated f15."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES optimizer for Rastrigin rotated in 5D with multi-start and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # Phase 1: Initial diverse sampling to find a good starting basin
        n_initial = min(30, budget // 10)
        for i in range(n_initial):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with adaptive restarts
        remaining = budget - calls_used
        if remaining > 20:
            best_x, best_f, calls_used = self._cma_es_restart(
                func, best_x, best_f, remaining, dim, rng, deadline, calls_used
            )

        # Phase 3: Final local polish around the best point
        if time.time() < deadline and calls_used < budget:
            fine_radius = 0.05
            for i in range(min(15, budget - calls_used)):
                if time.time() > deadline:
                    break
                x = best_x + rng.normal(0, fine_radius / dim**0.5, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    fine_radius *= 0.5

        return (best_x, best_f)

    def _cma_es_restart(self, func, x0, f0, total_budget, dim, rng, deadline, calls_used):
        """CMA-ES with adaptive restarts: restart when progress stalls."""
        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        global_best_x = x0.copy()
        global_best_f = f0

        # Adaptive chunk sizing: start large for exploration, shrink for exploitation
        # First chunk is larger, subsequent chunks get smaller
        chunk_sizes = []
        remaining_budget = total_budget
        restart_idx = 0
        while remaining_budget > 15:
            if restart_idx == 0:
                chunk = min(100, max(40, remaining_budget // 3))
            elif restart_idx < 3:
                chunk = min(60, max(25, remaining_budget // 4))
            else:
                chunk = min(40, max(20, remaining_budget // 5))
            chunk_sizes.append(chunk)
            remaining_budget -= chunk
            restart_idx += 1
            if len(chunk_sizes) >= 10:
                break

        for restart, chunk_size in enumerate(chunk_sizes):
            if time.time() > deadline or calls_used >= self.budget:
                break

            # Determine starting point for this restart
            if restart == 0:
                x_start = x0.copy()
                f_start = f0
            else:
                # Decaying perturbation: large early (exploration), small late (exploitation)
                noise_scale = 1.0 / (restart + 1)
                x_start = global_best_x + rng.normal(0, noise_scale, dim)
                x_start = np.clip(x_start, -4.5, 4.5)
                f_start = func(x_start)
                calls_used += 1
                if f_start < global_best_f:
                    global_best_f = f_start
                    global_best_x = x_start.copy()

            # Run CMA-ES from x_start with chunk_size budget
            x_res, f_res, calls_used = self._cma_es_single(
                func, x_start, f_start, chunk_size, dim, rng, deadline, calls_used,
                lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu
            )

            if f_res < global_best_f:
                global_best_f = f_res
                global_best_x = x_res.copy()

        return (global_best_x, global_best_f, calls_used)

    def _cma_es_single(self, func, x0, f0, budget, dim, rng, deadline, calls_used,
                       lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu):
        """Run a single CMA-ES instance with given budget."""
        mean = x0.copy()
        sigma = 1.0
        best_x = x0.copy()
        best_f = f0

        # Initialize evolution path and covariance
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Cholesky factor
        try:
            L = np.linalg.cholesky(C)
            C_sqrt_inv = np.linalg.inv(L)
        except:
            C_sqrt_inv = np.eye(dim)

        # Track best value in this CMA-ES run for early termination
        run_best_f = f0
        no_improve_count = 0
        max_no_improve = 3 * lam  # Terminate if no improvement for 3 generations

        # CMA-ES loop
        while calls_used < self.budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lam):
                if time.time() > deadline or calls_used >= self.budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt_inv @ z)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                pop.append((x, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if fx < run_best_f:
                    run_best_f = fx
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if len(pop) < mu:
                break

            # Early termination if no improvement for many evaluations
            if no_improve_count >= max_no_improve:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            selected = pop[:mu]

            # Update mean
            new_mean = np.zeros(dim)
            for i, (x, fx) in enumerate(selected):
                new_mean += weights[i] * x
            mean = new_mean

            # Update sigma
            actual_sigma = np.sqrt(np.mean([
                np.sum((x - mean)**2) for x, _ in selected
            ])) / np.sqrt(dim)
            if actual_sigma > 1e-10:
                sigma *= math.exp(c_sigma / d_sigma * (actual_sigma / sigma - 1))

            # Update evolution paths
            mean_selected = np.mean([x for x, _ in selected], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (mean_selected - mean) / sigma
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (mean_selected - mean) / sigma

            # Update covariance matrix
            diff = np.zeros((dim, dim))
            for i, (x, fx) in enumerate(selected):
                d = (x - mean) / sigma
                diff += weights[i] * np.outer(d, d)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(dim)) + c_mu * diff
            C = (C + C.T) / 2

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
            except:
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

            # Check for convergence
            if sigma < 1e-10:
                break

        return (best_x, best_f, calls_used)
