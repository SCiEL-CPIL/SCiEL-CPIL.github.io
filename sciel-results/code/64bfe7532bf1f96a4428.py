"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES with full covariance adaptation for BBOB f22.
    Strategy: quick Halton scan, then sequential CMA-ES restarts from
    the best points found, each with full covariance matrix updates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def halton_sequence(index, dim):
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        # Phase 1: Halton scan (30 points)
        n_scan = min(30, budget // 15)
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Phase 2: Multi-start CMA-ES with full covariance adaptation
        def cma_es_full(x0, sigma0, max_evals):
            nonlocal calls, best_f, best_x

            if calls >= budget or time.time() > deadline:
                return

            pop_size = max(4, int(4 + 3 * np.log(dim)))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            max_generations = max(1, max_evals // pop_size)

            for it in range(max_generations):
                if calls >= budget or time.time() > deadline:
                    break

                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) * 1e-8

                # Generate population
                population = []
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    fx = eval_and_track(x)
                    population.append((x, fx, z))

                if len(population) == 0:
                    break

                population.sort(key=lambda t: t[1])

                # Update mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * population[i][0]
                x_new = np.clip(x_new, -5, 5)

                # Update ps
                y_mean = (x_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
                norm_ps = np.linalg.norm(ps)
                h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
                sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update pc
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

                # Update covariance C
                delta_hs = (1 - c_c) / (1 - c_c**2)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = population[i][2]
                    rankmu += weights[i] * np.outer(z_i, z_i)

                C = (1 + c_1 * (delta_hs - 1) - c_mu) * C + c_1 * (rank1 + delta_hs * (np.eye(dim) - C)) + c_mu * rankmu

                # Symmetrize and ensure positive semi-definite
                C = 0.5 * (C + C.T)
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                C = (eigenvectors * eigenvalues) @ eigenvectors.T

                x_mean = x_new

            return x_mean

        # Multi-start CMA-ES: run from best point, then from second-best, etc.
        # Track top points from scan
        scan_points = []
        for i in range(1, n_scan + 1):
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            scan_points.append(x)

        # Sort scan points by their fitness (we need to re-evaluate or track)
        # Actually, let's just do sequential CMA-ES restarts
        n_restarts = 3
        remaining = budget - calls

        if remaining > 20:
            # First CMA-ES: from best point, use 50% of remaining
            cma_budget_1 = int(remaining * 0.5)
            if cma_budget_1 > 10:
                cma_es_full(best_x.copy(), 1.5, cma_budget_1)

            remaining = budget - calls
            if remaining > 15:
                # Second CMA-ES: from best point with smaller sigma
                cma_budget_2 = int(remaining * 0.4)
                if cma_budget_2 > 10:
                    cma_es_full(best_x.copy(), 0.5, cma_budget_2)

            remaining = budget - calls
            if remaining > 10:
                # Third CMA-ES: explore around best with medium sigma
                cma_budget_3 = remaining
                if cma_budget_3 > 10:
                    cma_es_full(best_x.copy(), 1.0, cma_budget_3)

        return best_x, best_f

# EVOLVE-BLOCK-END
