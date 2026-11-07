"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: multi-start exploration + CMA-ES exploitation + Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start exploration to find promising basins
        # Use multiple random starts with small local search
        n_starts = 15
        best_x = None
        best_fx = np.inf
        calls = 0

        for s in range(n_starts):
            if not time_ok() or calls >= budget:
                break

            # Random start
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            calls += 1

            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()

            # Small local search from this start (5 steps)
            x_local = x0.copy()
            fx_local = fx0
            sigma_local = 0.5

            for _ in range(5):
                if not time_ok() or calls >= budget:
                    break
                y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx_local:
                    x_local, fx_local = y, fy
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.8

            if fx_local < best_fx:
                best_fx = fx_local
                best_x = x_local.copy()

        if best_x is None:
            return None, None

        # Phase 2: CMA-ES exploitation from the best point found
        # CMA-ES parameters
        lam = max(4, 4 + int(3 * np.log(dim)))  # Population size
        mu = lam // 2  # Number of individuals contributing to update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial parameters
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Run CMA-ES until budget or convergence
        cma_calls = 0
        max_cma_calls = budget - calls - 30  # Reserve some calls for refinement

        while cma_calls < max_cma_calls and time_ok() and calls < budget:
            # Generate population
            population = []
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break

                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                population.append((y, fy, z))

            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[1])

            # Update best
            if population[0][1] < best_fx:
                best_fx = population[0][1]
                best_x = population[0][0].copy()

            # Weighted recombination
            y_w = np.zeros(dim)
            for i in range(min(mu, len(population))):
                y_w += weights[i] * population[i][0]

            # Update mean
            x_mean_new = y_w.copy()

            # Update step size control (sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (L @ (y_w - x_mean))

            # Check for sigma update
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (y_w - x_mean) / sigma

            # Compute delta_heavy for h1
            delta_heavy = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (cma_calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            # Rank-one and rank-mu updates
            C = (1 + c_1 * delta_heavy - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + (1 - delta_heavy) * c_c * (2 - c_c) * C) \
                + c_mu / sigma ** 2 * np.sum([
                    weights[i] * np.outer(population[i][0] - x_mean, population[i][0] - x_mean)
                    for i in range(min(mu, len(population)))
                ])

            # Ensure C is positive semi-definite
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            if np.min(eigenvalues) < 1e-10:
                C = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T

            x_mean = x_mean_new
            cma_calls += 1

            # Convergence check
            if sigma < 1e-10:
                break

        # Phase 3: Rastrigin-aware refinement
        remaining = budget - calls
        if remaining > 10 and time_ok():
            # Try integer snapping for each coordinate
            for coord in range(dim):
                if not time_ok() or calls >= budget - 5:
                    break
                # Try rounding current coordinate to nearest integer
                x_int = best_x.copy()
                x_int[coord] = np.round(best_x[coord])
                x_int = np.clip(x_int, -5, 5)
                fx_int = func(x_int)
                calls += 1
                if fx_int < best_fx:
                    best_x, best_fx = x_int, fx_int

            # Coordinate-wise local search with small steps
            sigma_small = 0.05
            for _ in range(min(remaining - 5, 20)):
                if not time_ok() or calls >= budget - 2:
                    break
                y = np.clip(best_x + sigma_small * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x, best_fx = y, fy
                    sigma_small *= 1.1
                else:
                    sigma_small *= 0.85
                    if sigma_small < 1e-9:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
