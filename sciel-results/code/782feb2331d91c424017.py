"""CMA-ES with adaptive restarts, diverse initialization, and final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive restarts and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def cma_es(x_mean, sigma, budget_for_this_restart):
            """Run CMA-ES from given mean and sigma."""
            nonlocal best_x, best_f, calls_used, deadline, rng

            n = dim
            # Population size and selection
            lambda_ = max(4 * n, 10)
            mu = lambda_ // 2

            # Logarithmic weights
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Strategy parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            sigma = sigma
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))

            local_best_x = x_mean.copy()
            local_best_f = np.inf
            no_improve_count = 0
            max_no_improve = 15

            max_iters = budget_for_this_restart // lambda_
            for it in range(max_iters):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    xi = x_mean + sigma * y
                    fi = evaluate(xi)
                    population.append((fi, xi, y))

                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort by fitness
                population.sort(key=lambda t: t[0])

                # Track local best
                if population[0][0] < local_best_f:
                    local_best_f = population[0][0]
                    local_best_x = population[0][1].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Stagnation check: break if no improvement for many iterations
                if no_improve_count >= max_no_improve:
                    break

                # Extract mu best steps
                y_sorted = [population[i][2] for i in range(mu)]
                x_sorted = [population[i][1] for i in range(mu)]

                # Update mean: weighted average of mu best
                x_mean = sum(w * xi for w, xi in zip(weights, x_sorted))

                # Update sigma path
                y_w_mean = sum(w * yi for w, yi in zip(weights, y_sorted))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B.T @ y_w_mean

                # Update C path
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2.0 / (n + 1)) * chi_n else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * B.T @ y_w_mean

                # Update covariance
                rank_one = np.outer(p_c, p_c) - (1 - c_c) * C
                rank_mu = sum(w * np.outer(yi, yi) for w, yi in zip(weights, y_sorted))
                C = (1 + c_1 * c_mu - c_1 - c_1) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is symmetric and positive definite
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update eigen decomposition
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))

                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Convergence check
                if sigma < 1e-10:
                    break

            return local_best_x, local_best_f

        # Phase 1: Initial sampling - diverse points
        n_initial = min(10, budget)
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners
        init_points.append(np.array([5.0] * dim))
        init_points.append(np.array([-5.0] * dim))
        # Random points
        for i in range(n_initial - 3):
            init_points.append(rng.uniform(-4, 4, dim))

        for x_init in init_points:
            if calls_used >= budget or time.time() > deadline:
                break
            evaluate(x_init)

        # Phase 2: Multi-restart CMA-ES with adaptive budget
        # Start with best found point, then perturb
        restart_configs = []
        remaining_budget = budget - calls_used

        # First restart: moderate sigma from best point
        if remaining_budget > 20:
            restart_configs.append((int(remaining_budget * 0.4), 2.0, best_x.copy()))
            remaining_budget -= int(remaining_budget * 0.4)

        # Second restart: larger sigma, perturbed
        if remaining_budget > 20:
            x_perturb = best_x + rng.standard_normal(dim) * 2.0
            x_perturb = np.clip(x_perturb, -5, 5)
            restart_configs.append((int(remaining_budget * 0.3), 3.0, x_perturb))
            remaining_budget -= int(remaining_budget * 0.3)

        # Third restart: small sigma, fine-tuning
        if remaining_budget > 15:
            restart_configs.append((int(remaining_budget * 0.3), 0.5, best_x.copy()))
            remaining_budget -= int(remaining_budget * 0.3)

        # Fourth restart: very small sigma
        if remaining_budget > 10:
            restart_configs.append((remaining_budget, 0.1, best_x.copy()))

        for restart_budget, sigma0, x_init in restart_configs:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            if restart_budget < 10:
                break

            cma_es(x_init, sigma0, restart_budget)

        # Final polish: coordinate descent with decreasing step sizes
        if calls_used < budget and time.time() < deadline:
            polish_calls = min(budget - calls_used, 50)
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            for ss in step_sizes:
                if calls_used >= budget or time.time() > deadline:
                    break
                for coord in range(dim):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        if calls_used >= budget or time.time() > deadline:
                            break
                        x_trial = best_x.copy()
                        x_trial[coord] += direction * ss
                        f_trial = evaluate(x_trial)
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
