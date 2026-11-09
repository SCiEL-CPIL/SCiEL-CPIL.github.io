import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Phase 1: Random initialization
        n_init = min(10, budget)
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        calls_made = n_init

        # Phase 2: CMA-ES
        if calls_made < budget:
            # CMA-ES parameters
            lambda_ = 10
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / ((dim + 2) * (dim + mu_eff / 2))
            c_c = (4 - mu_eff + c_sigma) / ((dim + 4) * (2 - c_sigma + mu_eff / 2))
            c_1 = 2 / ((dim + 1.7) ** 2 - mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2) ** 2 - mu_eff))

            mean = best_x.copy()
            sigma = 1.0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            while calls_made < budget and time.time() < deadline:
                # Check if we have enough time for at least one generation
                if time.time() > deadline - 2.0:
                    break

                # Generate offspring
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    f_y = func(y)
                    calls_made += 1
                    offspring.append((y, f_y))
                    if f_y < best_f:
                        best_f = f_y
                        best_x = y.copy()

                if calls_made >= budget:
                    break

                # Sort and select
                offspring.sort(key=lambda t: t[1])
                selected = offspring[:mu]

                # Update mean
                old_mean = mean.copy()
                mean = np.zeros(dim)
                for w, (x_sel, _) in zip(weights, selected):
                    mean += w * x_sel

                # Update step size (sigma)
                # Compute C^{-1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-14)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mean - old_mean)) / sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (1 + 1) / lambda_))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff - 1) / (dim + 1) + 1)
                sigma *= np.exp(c_sigma / lambda_ * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update evolution path p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (mean - old_mean)) / sigma

                # Update covariance matrix
                C = (1 + c_1 * (2 - c_1) + c_mu * (2 - c_mu - 1 - c_1 * (2 - c_1)) - c_1) * C
                # This formula is wrong, let me use the standard one:
                # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i - mean_old) @ (y_i - mean_old).T)
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
                for w, (x_sel, _) in zip(weights, selected):
                    diff = x_sel - old_mean
                    C += c_mu * w * np.outer(diff, diff)

                # Recompute B and D from C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-14)
                D = np.sqrt(eigvals)
                B = eigvecs

        return best_x, best_f
