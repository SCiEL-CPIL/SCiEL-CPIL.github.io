"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer with CMA-ES for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def should_stop():
            return time.time() > deadline

        # Phase 1: Broad exploration to find promising regions
        n_restarts = 8
        sigma_explore = 2.5
        local_steps = 5

        for r in range(n_restarts):
            if should_stop() or calls_made >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls_made += 1

            # Local refinement from this starting point
            sigma = sigma_explore
            for _ in range(local_steps):
                if should_stop() or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.75

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if should_stop() or calls_made >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES for systematic optimization
        # CMA-ES parameters
        pop_size = 4 + int(np.floor(3 * np.log(dim)))  # ~11 for dim=5
        mu = pop_size // 2  # number of parents
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
        max_gens = (budget - calls_made) // pop_size
        for gen in range(max_gens):
            if should_stop() or calls_made >= budget:
                break

            # Generate population
            x_pop = np.zeros((pop_size, dim))
            f_pop = np.zeros(pop_size)

            for i in range(pop_size):
                if should_stop() or calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                x_pop[i] = y
                f_pop[i] = func(y)
                calls_made += 1

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
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            x_mean = x_new_mean

            # Stopping criterion: convergence
            if sigma < 1e-10 or np.all(np.abs(D) < 1e-10):
                break

        # Phase 3: Fine-tuning
        if not should_stop() and calls_made < budget:
            sigma = 0.02
            for _ in range(min(50, budget - calls_made)):
                if should_stop() or calls_made >= budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.15
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
