"""Optimizer for BBOB-style f15 (Rastrigin rotated) with CMA-ES + periodicity-aware restarts."""
import time
import numpy as np


class Optimizer:
    """CMA-ES with periodicity-aware restart mechanism for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Track best solution globally
        best_x = None
        best_f = np.inf

        # Number of evaluations used so far
        evals_used = 0

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights
        w = np.log(np.arange(1, lambda_ + 1) / np.arange(lambda_, 0, -1))
        w = w / w.sum()
        mu = int(np.floor(lambda_ / 2))
        w = w[:mu]
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / dim) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Restart parameters
        # Rastrigin has periodicity ~ 2*pi in each dimension
        # Periodicity-aware: if we detect stagnation, restart with larger sigma
        stagnation_threshold = 50  # number of evaluations without improvement
        restart_sigma_factor = 2.0

        def cma_es_run(x0, sigma0, max_evals, deadline):
            nonlocal best_x, best_f, evals_used
            nonlocal budget

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            # Evaluate initial point
            fx = func(x_mean)
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_mean.copy()

            local_best = fx
            local_best_x = x_mean.copy()
            stagnation_count = 0
            local_evals = 0

            while evals_used < max_evals and time.time() < deadline:
                # Generate population
                # Eigendecomposition for C
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    break

                # Generate candidates
                candidates = []
                for i in range(lambda_):
                    z = np.random.standard_normal(dim)
                    y = x_mean + sigma * (eigvecs @ (D * z))
                    y = np.clip(y, -5, 5)
                    candidates.append(y)

                # Evaluate population
                fvals = []
                for y in candidates:
                    if evals_used >= max_evals or time.time() >= deadline:
                        break
                    fy = func(y)
                    fvals.append(fy)
                    evals_used += 1
                    local_evals += 1

                if len(fvals) < mu:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                selected_idx = order[:mu]
                selected_f = [fvals[i] for i in selected_idx]
                selected_y = [candidates[i] for i in selected_idx]

                # Check for improvement
                min_f = min(fvals)
                if min_f < local_best:
                    local_best = min_f
                    local_best_x = candidates[int(np.argmin(fvals))].copy()
                    stagnation_count = 0
                    if local_best < best_f:
                        best_f = local_best
                        best_x = local_best_x.copy()
                else:
                    stagnation_count += len(fvals)

                # Update mean
                x_new = np.sum([w[i] * selected_y[i] for i in range(mu)], axis=0)
                delta_x = x_new - x_mean

                # Update step size control variable
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C ** -0.5) @ delta_x / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (local_evals + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

                # Update covariance matrix
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_x / sigma
                C = (1 + c_1 * (hsig - 1) - c_1) * C + c_1 * np.outer(pc, pc) + c_mu * sum(
                    w[i] * np.outer(selected_y[i] - x_mean, selected_y[i] - x_mean) / sigma ** 2 for i in range(mu)
                )

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update mean
                x_mean = x_new

                # Periodicity-aware restart: if stagnation detected, restart with larger sigma
                if stagnation_count > stagnation_threshold:
                    # Restart: center on best incumbent, larger sigma
                    x_mean = local_best_x.copy()
                    sigma = max(sigma * restart_sigma_factor, 1.0)
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    stagnation_count = 0

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-4, 10.0)

            return local_best, local_best_x

        # Multi-restart strategy: run multiple CMA-ES instances with different initializations
        # First, do a quick random search to get a reasonable starting point
        rng = np.random.default_rng(42)

        # Initial random sampling to find a good starting point
        n_initial = min(20, budget // 10)
        best_init_x = None
        best_init_f = np.inf
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            evals_used += 1
            if fx < best_init_f:
                best_init_f = fx
                best_init_x = x.copy()
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Run CMA-ES from the best initial point
        remaining = budget - evals_used
        if remaining > 10:
            cma_es_run(best_init_x, 2.0, evals_used + remaining, deadline)

        # If we have budget left, try a few more restarts from different points
        remaining = budget - evals_used
        if remaining > 30 and time.time() < deadline:
            # Try restarts from perturbed best solution
            n_restarts = min(3, remaining // 30)
            for r in range(n_restarts):
                if evals_used >= budget or time.time() >= deadline:
                    break
                # Perturb the best solution
                perturbation = 3.0 * rng.standard_normal(dim)
                x0 = np.clip(best_x + perturbation, -5, 5)
                remaining = budget - evals_used
                if remaining > 10:
                    cma_es_run(x0, 1.5, evals_used + remaining, deadline)

        # Final safety: ensure we return something
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            evals_used += 1

        return best_x, best_f
