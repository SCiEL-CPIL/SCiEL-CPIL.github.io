"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for rotated Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = max(4 + int(3 * np.log(dim)), 6)  # population size ~10 for dim=5
        mu = lam // 2  # number of recombination weights
        # Logarithmic weights
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x_mean = np.zeros(dim)  # start at origin (shifted minimizer is in [-4,4]^5)
        sigma = 2.0  # initial step size
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Cache evaluations
        eval_count = 0
        best_x = x_mean.copy()
        best_f = func(x_mean)
        eval_count += 1

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Evaluate initial mean if not done (already done above)

        gen = 0
        while eval_count < budget:
            if time.time() > deadline:
                break
            gen += 1

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # x = x_mean + sigma * B @ D * z
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                offspring.append(x_i)

            # Evaluate offspring
            f_values = [evaluate(x_i) for x_i in offspring]

            # Sort by fitness
            order = np.argsort(f_values)
            x_selected = [offspring[i] for i in order[:mu]]
            f_selected = [f_values[i] for i in order[:mu]]

            # Weighted mean
            x_new_mean = sum(w[i] * x_selected[i] for i in range(mu))

            # Update step-size control vector ps
            inv_sqrt_C = C_sqrt_inv.T  # 1/sqrt(C)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                inv_sqrt_C @ (x_new_mean - x_mean) / sigma
            )
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (
                1.4 + 2 / (dim + 1)
            ) * chi_n

            # Update translation control vector pc
            pc = (1 - c_c) * pc + float(hsig) * np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            delta_hsig = (1 - float(hsig)) * c_c * (2 - c_c)
            outer = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
            C = (
                (1 + c_1 * delta_hsig - c_1 - c_mu) * C
                + c_1 * (outer / sigma ** 2 + delta_hsig * np.outer(pc, pc) - np.eye(dim))
                + c_mu * sum(w[i] * np.outer(x_selected[i] - x_mean, x_selected[i] - x_mean) for i in range(mu)) / sigma ** 2
            )

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Recompute eigenvalue decomposition periodically
            if gen % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)

            # Update sigma
            sigma *= np.exp(
                c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
