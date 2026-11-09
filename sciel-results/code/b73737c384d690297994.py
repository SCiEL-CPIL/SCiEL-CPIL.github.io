"""Optimized program for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial random sampling to find a good starting point
        # Use 20% of budget for initial exploration
        n_init = max(10, budget // 5)
        best_x = None
        best_f = np.inf

        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES refinement from the best point
        # Use remaining budget
        remaining = budget - n_init
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        sigma = 1.0  # Initial step size
        mean = best_x.copy()
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        calls_made = n_init

        for gen in range(remaining // pop_size + 1):
            if time.time() > deadline:
                break

            # Generate population
            archive_x = []
            archive_f = []

            for i in range(pop_size):
                if time.time() > deadline or calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * C_sqrt @ z
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                archive_x.append(x)
                archive_f.append(fx)

                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()

            if len(archive_x) < mu:
                break

            # Sort by fitness
            indices = np.argsort(archive_f)
            archive_x = [archive_x[i] for i in indices]
            archive_f = [archive_f[i] for i in indices]

            # Update mean
            old_mean = mean.copy()
            mean = sum(weights[i] * archive_x[i] for i in range(mu))
            mean = np.clip(mean, -5, 5)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt_inv @ (mean - old_mean))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - n_init + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= 1.5

            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (C_sqrt_inv @ (mean - old_mean))
            delta_sigma = (np.linalg.norm(p_c) ** 2 / (1 - (1 - c_c) ** (2 * (calls_made - n_init + 1))) - chi_n ** 2) * c_c

            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_sigma * C)
            for i in range(mu):
                y_i = C_sqrt_inv @ (archive_x[i] - old_mean)
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Recompute C_sqrt using eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(mean - old_mean) < 1e-12):
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
