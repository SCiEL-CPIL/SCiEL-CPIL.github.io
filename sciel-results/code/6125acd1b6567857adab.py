"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: broad exploration with (1+1)-ES, then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def should_stop():
            return time.time() > deadline or calls >= budget

        # Phase 1: Multi-start (1+1)-ES for broad exploration
        # Use 40% of budget for exploration
        phase1_budget = int(budget * 0.40)
        n_restarts = 5
        per_restart = max(10, phase1_budget // n_restarts)

        for r in range(n_restarts):
            if should_stop():
                break

            # Initialize: first restart random, others near best
            if best_x is not None and r > 0:
                x = best_x + 1.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 2.0
            success_count = 0
            window = 10

            for i in range(per_restart - 1):
                if should_stop():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()

                # Adapt sigma based on success rate over window
                if (i + 1) % window == 0:
                    rate = success_count / window
                    if rate > 0.5:
                        sigma *= 1.3
                    elif rate < 0.3:
                        sigma *= 0.7
                    success_count = 0

                # Keep sigma in reasonable range
                sigma = max(0.01, min(sigma, 10.0))

            if should_stop():
                break

        # Phase 2: CMA-ES for systematic refinement
        if should_stop() or best_x is None:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 4 + int(np.floor(3 * np.log(dim)))  # ~11 for dim=5
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.arange(mu)
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 2.0 / (mu_eff + 1.3))) / ((dim + 2) ** 2 + 2))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES loop
        remaining_budget = budget - calls
        max_gens = remaining_budget // pop_size

        for gen in range(max_gens):
            if should_stop():
                break

            # Generate population
            x_pop = np.zeros((pop_size, dim))
            f_pop = np.zeros(pop_size)

            for i in range(pop_size):
                if should_stop():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                x_pop[i] = y
                f_pop[i] = func(y)
                calls += 1

            # Sort by fitness
            order = np.argsort(f_pop)
            f_sorted = f_pop[order]
            x_sorted = x_pop[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Compute new mean
            x_new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update step size (Cumulative Step Adaptation)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_new_mean - x_mean)))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update evolution path
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cs * (2 - cs) * C)
            for i in range(mu):
                diff = (x_sorted[i] - x_mean) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            B, D = eigvecs, np.sqrt(eigvals)

            # Update mean
            x_mean = x_new_mean

            if should_stop():
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
