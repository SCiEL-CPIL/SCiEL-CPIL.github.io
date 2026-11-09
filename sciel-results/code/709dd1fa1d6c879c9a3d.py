"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with portfolio management for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_fx = np.inf
        eval_count = 0

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of parents

        def cma_es_run(x0, sigma0, max_evals):
            nonlocal best_x, best_fx, eval_count

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            # Learning rates
            alpha_mu = 2.0 / (n + 4)
            c_sigma = (alpha_mu + 2) / (n + 4)
            c_c = (4 + alpha_mu / c_sigma) / (n + 4.3)
            c_1 = 2 / ((n + 1.3)**2 + alpha_mu)
            c_mu = min(1 - c_1, 2 * (alpha_mu - 2 + 2 / alpha_mu) / ((n + 2)**2 + alpha_mu))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            ps = np.zeros(n)
            pc = np.zeros(n)
            local_best = x.copy()
            local_best_fx = np.inf
            local_evals = 0

            while local_evals < max_evals:
                if time.time() > deadline:
                    break

                # Generate population
                Y = []
                F = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    eval_count += 1
                    local_evals += 1
                    Y.append(y)
                    F.append(fy)
                    if fy < local_best_fx:
                        local_best = y.copy()
                        local_best_fx = fy
                    if fy < best_fx:
                        best_x = y.copy()
                        best_fx = fy

                Y = np.array(Y)
                F = np.array(F)

                # Sort by fitness
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]

                # Update mean with weighted recombination
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights**2)
                x_new = np.sum(weights[:, None] * Y[:mu], axis=0)
                x = x_new

                # Update step size
                d_sigma = c_sigma * (1 + 2 * min(mu_eff / (n + 1), 1))
                ps = (1 - d_sigma) * ps + d_sigma / chi_n * np.sqrt(mu_eff) * np.dot(weights, Y[:mu])
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update covariance matrix
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - d_sigma)**(2 + local_evals / 100))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + h_sigma * c_c * mu_eff * (x - x_new) / sigma
                delta_pc = (1 - h_sigma) * (1 - c_c) * (1 + c_c) * chi_n**2 / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_pc) + c_mu * np.sum(
                    weights[:, None, None] * np.outer(Y[:mu] - x, Y[:mu] - x), axis=0) / sigma**2

            return local_best, local_best_fx

        # Phase 1: Broad exploration with multiple CMA-ES starts
        # Try different initializations to find promising basins
        num_starts = 5
        evals_per_start = max(30, budget // num_starts // 2)

        for s in range(num_starts):
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break

            if s == 0:
                x0 = np.zeros(n)
            elif s == 1:
                x0 = rng.uniform(-3, 3, n)
            elif s == 2:
                x0 = rng.uniform(-4, 4, n)
            elif s == 3:
                # Try corners
                x0 = rng.choice([-4, 4], size=n) * rng.random(n)
            else:
                x0 = rng.uniform(-5, 5, n)

            sigma0 = 2.0 if s < 3 else 1.0

            cma_es_run(x0, sigma0, evals_per_start)

        # Phase 2: Exploit best region found
        remaining = budget - eval_count
        if remaining > 20 and time.time() < deadline:
            # Run a focused CMA-ES from the best point
            cma_es_run(best_x.copy(), 0.5, remaining)

        # Phase 3: Fine-tuning with small steps
        remaining = budget - eval_count
        if remaining > 10 and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                eval_count += 1
                if fy < fx:
                    x, fx = y, fy
                    if fy < best_fx:
                        best_x = y.copy()
                        best_fx = fy
                    sigma *= 0.95
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
