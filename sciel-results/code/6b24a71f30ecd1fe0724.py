"""Optimized BBOB f6 solver using CMA-ES with restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart strategy for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        # Try multiple restarts with different initializations
        # Use a mix of strategies: CMA-ES with varying population sizes

        rng = np.random.default_rng(42)

        # Strategy 1: CMA-ES with moderate population
        # For 500 budget in 5D, use lambda ~ 4+floor(3*ln(n)) = 4+13 = 17, but that's expensive
        # Let's use a smaller population for more iterations: lambda=9
        # Each iteration uses lambda+1 evaluations (lambda offspring + 1 parent re-eval)
        # Actually CMA-ES typically evaluates lambda offspring per generation
        # With budget 500, we can do about 500/lambda generations

        # Let's implement a simple CMA-ES
        def cma_es(x0, sigma0, lam, max_iter, time_check):
            nonlocal total_calls, best_x, best_f

            n = dim
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Parameters
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            def update_covariance(psi, w_psi):
                nonlocal C, C_inv_sqrt, C_sqrt
                C = (1 - c_1 - c_mu) * C + c_1 * (2 - mu_eff) * np.outer(psi, psi)
                for i in range(len(w_psi)):
                    C += w_psi[i] * np.outer(psi[i], psi[i])
                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Update square root decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Evaluate initial point
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            for gen in range(max_iter):
                if total_calls + lam > budget:
                    break
                if time.time() > time_check:
                    break

                # Sample offspring
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                f_vals = []
                for y in offspring:
                    if time.time() > time_check:
                        break
                    fy = func(y)
                    total_calls += 1
                    f_vals.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(f_vals) < lam:
                    break

                # Sort by fitness
                order = np.argsort(f_vals)
                sorted_offspring = [offspring[i] for i in order]
                sorted_fvals = [f_vals[i] for i in order]

                # Weighted recombination
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += w[i] * sorted_offspring[i]

                # Update sigma
                z = C_inv_sqrt @ (x_new - x)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * d_sigma * z
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / np.sqrt(1 - c_sigma)) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff + 1)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

                # Update path and covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new - x)
                psi = [C_inv_sqrt @ (sorted_offspring[i] - x) for i in range(mu)]
                update_covariance(psi, w)

                x = x_new

                # Re-evaluate parent (optional, but helps with accuracy)
                # We skip re-evaluating parent to save budget

            return x, best_f

        # Multi-restart strategy
        # Divide budget among restarts
        # First restart: larger budget, later restarts: smaller
        restarts = [
            # (initial_x, sigma, lambda, max_generations)
        ]

        # Generate diverse starting points
        n_restarts = 4
        budget_per_restart = [
            int(budget * 0.4),
            int(budget * 0.25),
            int(budget * 0.2),
            int(budget * 0.15),
        ]
        # Ensure sum doesn't exceed budget
        while sum(budget_per_restart) > budget:
            budget_per_restart[-1] -= 1
            if budget_per_restart[-1] <= 0:
                budget_per_restart.pop()

        for r_idx, bpr in enumerate(budget_per_restart):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            # Different initialization strategies
            if r_idx == 0:
                # Center + small perturbation
                x0 = np.zeros(dim) + rng.standard_normal(dim) * 0.5
                sigma0 = 1.0
                lam = 9
            elif r_idx == 1:
                # Random point
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.5
                lam = 7
            elif r_idx == 2:
                # Another random point
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 1.0
                lam = 7
            else:
                # Near the best found so far
                if best_x is not None:
                    x0 = best_x + rng.standard_normal(dim) * 0.3
                else:
                    x0 = rng.uniform(-2, 2, dim)
                sigma0 = 0.5
                lam = 5

            max_iter = max(1, bpr // lam)
            cma_es(x0, sigma0, lam, max_iter, deadline)

        # Final fine-tuning with (1+1)-ES if we have budget left
        if best_x is not None and total_calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - total_calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
