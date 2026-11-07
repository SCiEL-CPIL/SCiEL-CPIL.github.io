"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """CMA-ES with aggressive restarts for multimodal BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls = 0

        # CMA-ES parameters
        lambda_pop = 10
        mu = lambda_pop // 2  # number of parents
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 + 2 * min(w[0], 0.5) * (mu_eff - 1) / (n + 2)) / (
            (n + 1.3) ** 2 + mu_eff
        )
        c_mu = min(
            1 - c_1,
            2 * (
                (sqrt_mu_eff := np.sqrt(mu_eff)) * (w[0] + w[1] - 2) / (n + 2)
            ),
        )

        def run_cma(x0, sigma0, max_evals):
            nonlocal calls
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Generate initial population
            pop = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                # Whitened sample: C^(1/2) z. Use Cholesky.
                L = np.linalg.cholesky(C)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                pop.append((y, None))

            # Evaluate
            for i, (y, _) in enumerate(pop):
                fy = func(y)
                calls += 1
                pop[i] = (y, fy)

            # Sort
            pop.sort(key=lambda t: t[1])
            best_x = pop[0][0].copy()
            best_f = pop[0][1]

            # Update
            for i in range(mu):
                z = (pop[i][0] - x_mean) / (sigma * np.sqrt(np.diag(C)))
                # Actually need C^{-1/2}(x - x_mean)/sigma. Let's do it properly.
                # But for simplicity and speed, use the standard CMA-ES update.
                pass

            # Standard CMA-ES update
            x_parents = np.array([pop[i][0] for i in range(mu)])
            x_recentered = x_parents - x_mean
            # Transform to normal space
            # y_i = C^{-1/2} (x_i - x_mean) / sigma
            # We can compute C^{-1/2} using eigen decomposition or Cholesky inverse
            L = np.linalg.cholesky(C)
            # Solve L @ v = (x_parents.T - x_mean) / sigma
            # Actually, (x_parents - x_mean) is (mu, n). We want C^{-1/2} @ x_parents.T
            # Let's compute inv_L = np.linalg.inv(L)
            inv_L = np.linalg.inv(L)
            y_parents = inv_L @ (x_parents.T / sigma)  # (n, mu)

            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                inv_L @ (x_mean / sigma)
            )
            # Wait, the standard formula is:
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * hsig * C^{-1/2} * (x_new - x_old)/sigma
            # where x_new - x_old is the step. Let me re-derive.

            # Let's restart with a cleaner implementation.

            return best_x, best_f

        # Simplified robust CMA-ES
        def cma_es(x0, sigma0, max_evals):
            nonlocal calls
            x_mean = x0.astype(float).copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Precompute
            L = np.linalg.cholesky(C)

            # Generate and evaluate population
            def sample_and_eval():
                nonlocal calls
                pop = []
                for _ in range(lambda_pop):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    pop.append((y, fy))
                pop.sort(key=lambda t: t[1])
                return pop

            pop = sample_and_eval()
            best_x = pop[0][0].copy()
            best_f = pop[0][1]

            evals_used = lambda_pop
            no_improve = 0
            best_f_hist = best_f

            while evals_used < max_evals:
                # Update mean
                x_parents = np.array([pop[i][0] for i in range(mu)])
                x_mean_new = np.sum(w[:, None] * x_parents, axis=0)

                # Update covariance
                # Compute y_i = C^{-1/2} (x_i - x_mean) / sigma
                L = np.linalg.cholesky(C)
                inv_L = np.linalg.inv(L)
                y_parents = inv_L @ ((x_parents - x_mean).T / sigma)  # (n, mu)

                # Update ps
                hsig = (
                    np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (max_evals - evals_used + lambda_pop)))
                    < (1 + 2 / (n + 1)) * chi_n
                )
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff
                ) * inv_L @ ((x_mean_new - x_mean) / sigma)

                # Update pc
                pc = (
                    1 - c_c
                ) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma

                # Update C
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (
                    (1 + c_1 + c_mu - c_1 - c_c) * C
                    + c_1 * (n + 2 - c_1) * C
                    - c_1 * pc[:, None] @ pc[None, :]
                    + c_mu * np.sum(w[:, None, None] * y_parents[:, :, None] * y_parents[:, None, :], axis=1)
                    - c_1 * delta_hsig * pc[:, None] @ pc[None, :]
                )
                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)
                )
                sigma = np.clip(sigma, 1e-10, 1e10)

                x_mean = x_mean_new

                # Evaluate new population
                pop = sample_and_eval()
                evals_used += lambda_pop

                if pop[0][1] < best_f - 1e-12:
                    best_f = pop[0][1]
                    best_x = pop[0][0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart condition
                if no_improve >= 20 or sigma < 1e-8:
                    break

            return best_x, best_f

        # Multi-start CMA-ES
        best_x_overall = None
        best_f_overall = np.inf

        # Initial exploration with a few random starts
        n_starts = 5
        evals_per_start = (total_budget - n_starts) // n_starts
        evals_per_start = max(evals_per_start, 50)

        for s in range(n_starts):
            if calls >= total_budget:
                break

            # Random start
            x0 = rng.uniform(-5, 5, n)
            sigma0 = 2.0

            remaining = total_budget - calls
            if remaining < lambda_pop:
                break

            max_evals = min(evals_per_start, remaining)
            # Ensure we can at least do one full population
            if max_evals < lambda_pop:
                max_evals = remaining

            x_best, f_best = cma_es(x0, sigma0, max_evals)

            if f_best < best_f_overall:
                best_f_overall = f_best
                best_x_overall = x_best

        # Use remaining budget with the best found
        if calls < total_budget and best_x_overall is not None:
            remaining = total_budget - calls
            if remaining >= lambda_pop:
                x_best, f_best = cma_es(best_x_overall, 1.0, remaining)
                if f_best < best_f_overall:
                    best_f_overall = f_best
                    best_x_overall = x_best

        if best_x_overall is None:
            best_x_overall = np.zeros(n)
            best_f_overall = func(best_x_overall)
            calls += 1

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
