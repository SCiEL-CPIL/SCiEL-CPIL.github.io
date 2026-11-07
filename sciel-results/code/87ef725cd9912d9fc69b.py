"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 300s limit

        best_x = np.array([0.0] * dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
            except Exception:
                val = np.inf
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def cma_es(x0, f0, sigma0, max_evals, rng):
            nonlocal calls_made, best_f, best_x, deadline

            dim = self.dim
            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim))
            lambda_ = pop_size

            # Step size control parameters
            c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (dim + 1)) - 1) + c_sigma

            # Covariance matrix adaptation parameters
            mu_w = lambda_ // 2  # number of weights
            w = np.log(np.arange(1, mu_w + 1))
            w = w / w.sum()
            w = np.concatenate([w, np.zeros(lambda_ - mu_w)])
            mu_eff = 1.0 / np.sum(w[:mu_w] ** 2)

            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            evaluations_in_this_run = 0
            history = []

            while evaluations_in_this_run < max_evals:
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    # Transform z using B and D
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    offspring.append(x_new)

                # Evaluate offspring
                f_offspring = np.array([safe_call(x) for x in offspring])
                evaluations_in_this_run += lambda_

                # Sort by fitness
                order = np.argsort(f_offspring)
                x_sorted = np.array([offspring[i] for i in order])
                f_sorted = f_offspring[order]

                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()

                # Update mean
                x_new_mean = np.sum(w[:, np.newaxis] * x_sorted[:mu_w], axis=0)

                # Update step size (CSA)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * (x_sorted[0] - x_mean) / sigma))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update evolution paths
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

                # Update covariance matrix
                delta_hc = (1 + hsig * (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (evaluations_in_this_run + 1) // lambda_)) - 1)) if False else 1.0
                # Simplified: just use standard update
                v = (x_sorted[:mu_w] - x_mean) / sigma
                C_old = (B * D) @ (B * D).T
                C_new = (1 - c_1 - c_mu) * C_old + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * np.eye(dim))
                for i in range(mu_w):
                    C_new += c_mu * w[i] * np.outer(v[i], v[i])

                # Eigen decomposition
                try:
                    eigenvalues, B = np.linalg.eigh(C_new)
                    eigenvalues = np.maximum(eigenvalues, 1e-15)
                    D = np.sqrt(eigenvalues)
                    B = B  # B is now the eigenvectors
                except np.linalg.LinAlgError:
                    # Fallback: reset
                    B = np.eye(dim)
                    D = np.ones(dim)

                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5.0, 5.0)

                # Termination check: if sigma is too small or no progress
                if sigma < 1e-8:
                    break

            return x_mean, f_sorted[0] if len(f_sorted) > 0 else np.inf

        # Main loop: multiple restarts
        rng_master = np.random.default_rng(42)

        # First, do a coarse global search to find a good starting point
        n_init = min(20, budget // 10)
        candidates = []
        for _ in range(n_init):
            x = rng_master.uniform(-4, 4, dim)
            fx = safe_call(x)
            candidates.append((fx, x))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Use the best candidate as the starting point for CMA-ES
        if candidates:
            f0, x0 = candidates[0]
        else:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)

        # Run CMA-ES with the remaining budget
        remaining = budget - calls_made
        if remaining > 0:
            cma_es(x0, f0, 1.0, remaining, rng_master)

        return best_x, best_f
# EVOLVE-BLOCK-END
