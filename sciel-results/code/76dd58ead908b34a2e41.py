"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multiple restarts for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill

        best_x = None
        best_f = np.inf

        dim = self.dim

        def cma_es(x0, sigma0, max_evals, rng):
            nonlocal best_x, best_f
            nonlocal deadline

            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(dim))
            mu = lambda_ // 2
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mu_eff = 1. / np.sum(weights ** 2)

            c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
            c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
            c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) - (dim + 2)) / ((dim + 2) ** 2 + lambda_))
            c_c = (c_mu + 2) / (dim + c_mu)
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            C = np.eye(dim)

            evals_used = 0
            gen = 0

            while evals_used < max_evals:
                if time.time() > deadline - 5:
                    break

                # Generate lambda samples
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    fy = func(y)
                    offspring.append((y, fy))
                    evals_used += 1
                    if time.time() > deadline - 5:
                        break

                if len(offspring) < lambda_:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])

                # Update best
                if offspring[0][1] < best_f:
                    best_f = offspring[0][1]
                    best_x = offspring[0][0].copy()

                # Weighted mean
                y_mean = np.zeros(dim)
                for i in range(mu):
                    y_mean += weights[i] * (offspring[i][0] - x)
                y_mean /= np.sum(weights[:mu])

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * y_mean))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2. / (dim + 1)) * chi_n:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

                # Update covariance matrix
                delta_hs = (np.linalg.norm(ps) > (1.4 + 2. / (dim + 1)) * chi_n)
                C = (1 + c_1 * delta_hs * (c_c + 1) - c_1) * C
                for i in range(mu):
                    yi = offspring[i][0] - x
                    C += c_mu * np.outer(yi, yi)
                C += c_1 * delta_hs * (np.outer(pc, pc) - C)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update B and D from C
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-12)
                    D = np.sqrt(D2)
                except np.linalg.LinAlgError:
                    D2 = np.ones(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 1e6)

                # Update mean
                x = x + y_mean

                gen += 1

            return x, best_f

        # Multiple restarts for multi-modal f6
        rng = np.random.default_rng(42)
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break

            # Different starting points for each restart
            if r == 0:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 1.5
                max_evals = budget_per_restart
            elif r == 1:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 2.0
                max_evals = budget_per_restart
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.5
                max_evals = self.budget - r * budget_per_restart

            x_res, f_res = cma_es(x0, sigma0, max_evals, rng)

            if time.time() > deadline - 15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
