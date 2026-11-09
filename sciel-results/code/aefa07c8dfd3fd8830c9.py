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
        n_initial = min(25, budget // 10)
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
            for i in range(min(10, budget - calls_used)):
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

        # Restart strategy: run CMA-ES for a chunk of budget, check progress, restart if stalled
        # Use smaller chunks for more frequent restarts in the remaining budget
        chunk_size = min(60, max(25, total_budget // 6))
        max_restarts = max(3, total_budget // chunk_size)

        for restart in range(max_restarts):
            if time.time() > deadline or calls_used >= self.budget:
                break

            # Determine starting point for this restart
            if restart == 0:
                x_start = x0.copy()
                f_start = f0
            else:
                # Perturb the global best with random noise, increasing perturbation with restarts
                # Also try jumping to a random location for diversity
                if restart % 2 == 0:
                    # Large jump to explore new region
                    x_start = rng.uniform(-4.0, 4.0, dim)
                else:
                    # Moderate perturbation around best
                    noise_scale = 0.8 + 0.3 * restart
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

            # Check for stagnation: if no improvement in this chunk, the restart logic handles it
            # by perturbing the best point more aggressively

        return (global_best_x, global_best_f, calls_used)

    def _cma_es_single(self, func, x0, f0, budget, dim, rng, deadline, calls_used,
                       lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu):
        """Run a single CMA-ES instance with given budget."""
        mean = x0.copy()
        sigma = 0.5  # Start with smaller sigma for faster convergence
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
        max_no_improve = 4 * lam  # Terminate if no improvement for 4 generations

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

            # Save old mean for path updates
            old_mean = mean.copy()

            # Update mean
            new_mean = np.zeros(dim)
            for i, (x, fx) in enumerate(selected):
                new_mean += weights[i] * x
            mean = new_mean

            # Update sigma using proper CMA-ES formula
            # Use the norm of the mean shift relative to sigma
            mean_shift = mean - old_mean
            actual_sigma = np.linalg.norm(mean_shift) / np.sqrt(mu_eff)
            if actual_sigma > 1e-10:
                sigma *= math.exp(c_sigma / d_sigma * (actual_sigma / sigma - 1))
            sigma = min(sigma, 10.0)  # Cap sigma to avoid divergence

            # Update evolution paths (using old_mean as reference)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (mean - old_mean) / sigma
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (mean - old_mean) / sigma

            # Update covariance matrix
            diff = np.zeros((dim, dim))
            for i, (x, fx) in enumerate(selected):
                d = (x - old_mean) / sigma
                diff += weights[i] * np.outer(d, d)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(dim)) + c_mu * diff
            C = (C + C.T) / 2

            # Ensure positive definiteness
            C[np.diag_indices(dim)] += 1e-10

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
            except:
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

            # Check for convergence
            if sigma < 1e-8:
                break

        return (best_x, best_f, calls_used)
