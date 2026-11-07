"""Rewritten optimizer for BBOB f19 (Griewank-Rosenbrock) with CMA-ES + restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != dim:
                x = np.resize(x, dim)
            f = float(func(x))
            calls_made += 1
            return f

        def run_cma_es(x0, sigma0, budget_alloc, rng_local):
            """Run CMA-ES with given initial point, sigma, and budget."""
            nonlocal best_x, best_f, calls_made

            n = dim
            # CMA-ES parameters
            lambda_ = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_local_x = x_mean.copy()
            best_local_f = evaluate(x_mean)

            for gen in range(budget_alloc // lambda_ + 1):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng_local.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5.0, 5.0)
                    offspring.append(y)

                # Evaluate
                offspring_f = []
                for y in offspring:
                    fy = evaluate(y)
                    offspring_f.append(fy)
                    if fy < best_local_f:
                        best_local_f = fy
                        best_local_x = y.copy()

                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort
                idx = np.argsort(offspring_f)
                selected = [offspring[i] for i in idx[:mu]]
                selected_f = [offspring_f[i] for i in idx[:mu]]

                # Update best
                for i in range(mu):
                    if selected_f[i] < best_local_f:
                        best_local_f = selected_f[i]
                        best_local_x = selected[i].copy()

                # Update mean
                x_new_mean = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)

                # Update step size
                y_mean = (x_new_mean - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y_mean))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance
                delta_sig = (1 - c_sigma) if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n else 0
                pc = (1 - c_c) * pc + c_c * (y_mean / sigma if sigma > 0 else np.zeros(n))
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (selected[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_sig) * np.eye(n)) + c_mu * rank_mu

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize
                C += 1e-10 * np.eye(n)

                # Eigendecomposition
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-12)
                    D = np.sqrt(D_sq)
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                    D = np.ones(n)

                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5.0, 5.0)

                if sigma < 1e-10:
                    break

            return best_local_x, best_local_f

        # Strategy: multiple restarts with CMA-ES
        # Phase 1: broad exploration with several restarts
        # Phase 2: refine best found

        # Initial broad search
        num_initial_restarts = 5
        budget_per_restart = max(20, budget // (num_initial_restarts + 2))

        for r in range(num_initial_restarts):
            if time.time() > deadline or calls_made >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            sigma0 = 1.5
            bx, bf = run_cma_es(x0, sigma0, budget_per_restart, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 2: refine around best found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 30:
                # Several refinements around best
                num_refine = min(4, remaining // 40)
                for r in range(num_refine):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    x0 = best_x + rng.normal(0, 0.5, dim)
                    x0 = np.clip(x0, -5, 5)
                    sigma0 = 0.3
                    alloc = min(remaining - calls_made, 80)
                    if alloc < 10:
                        break
                    bx, bf = run_cma_es(x0, sigma0, alloc, rng)
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()

        # Final local refinement
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                x0 = best_x.copy()
                sigma0 = 0.05
                alloc = min(remaining, 50)
                bx, bf = run_cma_es(x0, sigma0, alloc, rng)
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
